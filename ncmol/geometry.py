"""Integrated geometric encoder, adapted from LiTEN-FF (MIT).

Copyright (c) 2025 lingcon01. See licenses/LiTEN-FF-MIT.txt.
Parameter names retain compatibility with the trained NCMol checkpoints.
"""
import math
import torch
from torch import Tensor, nn
from torch_geometric.nn.inits import glorot


def scatter(src, index, dim=0, reduce="sum", dim_size=None):
    """Sum-only native scatter used by geometric messages."""
    if dim != 0 or reduce != "sum":
        raise ValueError("Geometric scatter requires dim=0 and sum reduction")
    size = dim_size if dim_size is not None else int(index.max()) + 1
    out = src.new_zeros((size,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


class CosineCutoff(nn.Module):

    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()

        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs

class VectorNormalization(nn.Module):
    def __init__(self, hidden_channels, norm_type="norm"):
        super(VectorNormalization, self).__init__()

        self.hidden_channels = hidden_channels
        self.eps = 1e-12

        self.dist_norm = nn.LayerNorm(hidden_channels)

        if norm_type == "norm":
            self.norm = self.vec_norm
        else:
            self.norm = self.none_norm

        self.reset_parameters()

    def reset_parameters(self):
        self.dist_norm.reset_parameters()

    @torch.jit.export
    def none_norm(self, vec):
        return vec

    @torch.jit.export
    def vec_norm(self, vec):

        eps = self.eps

        dist = torch.norm(vec, dim=1, keepdim=True)  # (num_atoms, 1, hidden_channels)
        if torch.all(dist == 0):
            return torch.zeros_like(vec)

        # 方向向量
        radial = vec / dist.clamp(min=eps)

        # 每个样本的 hidden_channel 维度上做 min-max 归一化
        min_val = torch.min(dist, dim=-1, keepdim=True)[0]
        max_val = torch.max(dist, dim=-1, keepdim=True)[0]
        dist_norm = (dist - min_val) / (max_val - min_val).clamp(min=eps)

        norm_vec = dist_norm * radial

        return norm_vec

    def forward(self, vec):

        vec = self.norm(vec)

        return vec


class GeometricEdges(nn.Module):
    def __init__(self, cutoff, max_neighbors=32, loop=True):
        """
        Distance module for computing pairwise distances in molecular graphs.

        Args:
            cutoff (float): Maximum distance threshold for edge connections.
            max_neighbors (int, optional): Maximum number of neighbors per node. Default is 10.
            include_self (bool, optional): Whether to include self-loops. Default is True.
        """
        super(GeometricEdges, self).__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.loop = loop

    def forward(self, positions, edge_indices, shift):
        """
        Computes edge index, edge distances, and edge vectors.

        Args:
            positions (torch.Tensor): Atomic positions of shape (num_atoms, 3).
            batch (torch.Tensor): Batch indices of shape (num_atoms,).
            edge_indices (torch.Tensor, optional): Precomputed edge indices of shape (2, num_edges).

        Returns:
            edge_indices (torch.Tensor): Edge indices of shape (2, num_edges).
            edge_distances (torch.Tensor): Computed Euclidean distances for edges.
            edge_vectors (torch.Tensor): Edge vectors between connected atoms.
        """
        # Compute edge vectors
        row, col = edge_indices[0], edge_indices[1]

        if shift is not None:
            edge_vectors = positions[row] - positions[col] - shift
        else:
            edge_vectors = positions[row] - positions[col]

        mask = (row != col).unsqueeze(-1).to(positions.dtype)  # [E,1]
        norms = torch.norm(edge_vectors, dim=-1, keepdim=True) * mask
        edge_vectors = edge_vectors / torch.where(mask == 1, norms, 1.0)
        edge_distances = norms.squeeze(-1)

        return edge_indices, edge_distances, edge_vectors


class ExponentialRadialBasis(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=False):
        super(ExponentialRadialBasis, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)



class GeometricInteraction(nn.Module):
    def __init__(self, num_heads, hidden_channels, activation, cutoff, update_edge, vec_norm):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.update_edge = update_edge
        self.vec_norm = vec_norm
        self.eps = 1e-8

        self.layernorm = nn.LayerNorm(hidden_channels)

        self.act = activation
        self.alpha = torch.nn.Parameter(torch.Tensor(1, self.num_heads, self.head_dim))

        self.cutoff = CosineCutoff(cutoff)

        self.vec_linear = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.cross_linear = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.node_linear = nn.Linear(hidden_channels, hidden_channels)
        self.edge_linear = nn.Linear(hidden_channels, hidden_channels)
        self.part_linear1 = nn.Linear(hidden_channels, hidden_channels * 2)
        self.part_linear2 = nn.Linear(hidden_channels, hidden_channels * 3)

        self.f_linear = nn.Linear(hidden_channels, hidden_channels)

        self.vecnorm = VectorNormalization(hidden_channels, norm_type="norm")

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        glorot(self.alpha)
        nn.init.xavier_uniform_(self.node_linear.weight)
        self.node_linear.bias.data.zero_()
        nn.init.xavier_uniform_(self.edge_linear.weight)
        self.edge_linear.bias.data.zero_()
        nn.init.xavier_uniform_(self.part_linear1.weight)
        self.part_linear1.bias.data.zero_()
        nn.init.xavier_uniform_(self.part_linear2.weight)
        self.part_linear2.bias.data.zero_()
        nn.init.xavier_uniform_(self.cross_linear.weight)
        nn.init.xavier_uniform_(self.vec_linear.weight)
        nn.init.xavier_uniform_(self.f_linear.weight)
        self.f_linear.bias.data.zero_()

    def edge_update(self, node_vector, edge_index, edge_vector, edge_feats):
        i, j = edge_index[0], edge_index[1]

        node_vector = self.cross_linear(node_vector)

        vec_cross_i = torch.cross(node_vector[i], edge_vector.unsqueeze(-1), dim=1)
        vec_cross_j = torch.cross(node_vector[j], edge_vector.unsqueeze(-1), dim=1)
        sum_phi = torch.sum(vec_cross_i * vec_cross_j, dim=1)

        delta_edge_feats = self.act(self.f_linear(edge_feats)) * sum_phi

        return delta_edge_feats

    def message(self, node_scalar, node_vector, edge_index, dist, edge_vector, edge_feats):
        i, j = edge_index[0], edge_index[1]
        edge_feats = self.act(self.edge_linear(edge_feats)).reshape(-1, self.num_heads, self.head_dim)
        node_scalar = self.node_linear(node_scalar).reshape(-1, self.num_heads, self.head_dim)
        n_nodes = len(node_scalar)
        attn = (node_scalar[i] + node_scalar[j] + edge_feats)
        attn = self.act(attn) * self.alpha
        attn = attn.sum(dim=-1) * self.cutoff(dist).unsqueeze(1)

        node_scalar = node_scalar[j] * edge_feats
        node_scalar = (node_scalar * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        node_sca1, node_sca2 = torch.split(self.act(self.part_linear1(node_scalar)), self.hidden_channels, dim=1)
        node_vector = node_vector[j] * node_sca1.unsqueeze(1) + node_sca2.unsqueeze(
            1) * edge_vector.unsqueeze(2)

        node_scalar = scatter(node_scalar, i, dim=0, reduce="sum", dim_size=n_nodes)
        node_vector = scatter(node_vector, i, dim=0, reduce="sum", dim_size=n_nodes)

        return node_scalar, node_vector, attn

    def node_update(self, node_scalar, node_vector):

        node_vec1, node_vec2 = torch.split(self.vec_linear(node_vector), self.hidden_channels, dim=-1)
        vec_tri = torch.sum(node_vec1 * node_vec2, dim=1)

        norm_vec = torch.sqrt(torch.sum(node_vec2 ** 2, dim=-2) + 1e-8)
        vec_qua = norm_vec ** 3

        node_scalar = self.part_linear2(node_scalar)

        node_sca1, node_sca2, node_sca3 = torch.split(node_scalar, self.hidden_channels, dim=1)

        delta_scalar = (vec_qua + vec_tri) * node_sca1 + node_sca2
        delta_vector = node_vec1 * node_sca3.unsqueeze(1)

        return delta_scalar, delta_vector

    def forward(self, node_scalar, node_vector, edge_index, dist, edge_feats, edge_vector,
                update_edge: bool = True, vec_norm: bool = False):

        scalar_out = self.layernorm(node_scalar)

        if self.vec_norm:
            node_vector = normalize_vectors(node_vector, self.eps)

        scalar_out, vector_out, attn_out = self.message \
            (scalar_out, node_vector, edge_index, dist, edge_vector, edge_feats)

        if self.update_edge:
            delta_edge_feats = self.edge_update(node_vector, edge_index, edge_vector, edge_feats)

            edge_feats = edge_feats + delta_edge_feats

        node_scalar = node_scalar + scalar_out
        node_vector = node_vector + vector_out

        delta_scalar, delta_vector = self.node_update(node_scalar, node_vector)

        node_scalar = node_scalar + delta_scalar
        node_vector = node_vector + delta_vector

        return node_scalar, node_vector, edge_feats

@torch.jit.script
def normalize_vectors(vec: Tensor, eps: float) -> Tensor:

    dist = torch.sqrt(torch.sum(vec ** 2, dim=1) + eps).unsqueeze(1)

    dist = dist.clamp(min=eps)
    radial = vec / dist

    min_val = torch.min(dist, dim=-1, keepdim=True)[0]
    max_val = torch.max(dist, dim=-1, keepdim=True)[0]
    dist_norm = (dist - min_val) / (max_val - min_val).clamp(min=eps)

    norm_vec = dist_norm * radial

    return norm_vec



class GeometricEncoder(nn.Module):

    def __init__(self, num_heads=8, num_layers=6, hidden_channels=256, num_rbf=32,
                 cutoff=5.0, max_neighbors=32, max_z=100):
        super(GeometricEncoder, self).__init__()

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.activation = nn.SiLU()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.drop_rate = [0.0, 0.0, 0.1, 0.05, 0.05]

        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.connect = GeometricEdges(cutoff, max_neighbors=max_neighbors, loop=True)
        self.radial_fn = ExponentialRadialBasis(cutoff, num_rbf)

        self.Interaction = nn.ModuleList()

        for idx in range(num_layers):
            update_edge = idx < num_layers - 1
            vec_norm = idx > 0
            layer = GeometricInteraction(num_heads=num_heads, hidden_channels=hidden_channels, activation=self.activation,
                        cutoff=cutoff, update_edge=update_edge, vec_norm=vec_norm)
            self.Interaction.append(layer)

        self.out_norm = nn.LayerNorm(hidden_channels)


        self.edge_embedding = nn.Linear(num_rbf, hidden_channels)

        self.readout_energy = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            self.activation,
            nn.Linear(hidden_channels // 2, 1),
        )

        self.readout_force = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            self.activation,
            nn.Linear(hidden_channels // 2, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.radial_fn.reset_parameters()
        for layer in self.Interaction:
            layer.reset_parameters()

        self.out_norm.reset_parameters()
