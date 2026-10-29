from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .geometry import GeometricEncoder


def _pool_sum(x: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = x.new_zeros((size, x.shape[-1]))
    out.index_add_(0, index, x)
    return out


class SpatialEncoderRegressor(nn.Module):
    """3D equivariant encoder adapter used by NCMol."""

    def __init__(
        self, checkpoint: str | Path | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = GeometricEncoder()
        state = (torch.load(checkpoint, map_location="cpu", weights_only=False)
                 if checkpoint is not None else self.backbone.state_dict())
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        allowed_missing = {
            "readout_energy.0.weight", "readout_energy.0.bias",
            "readout_energy.2.weight", "readout_energy.2.bias",
            "readout_force.0.weight", "readout_force.0.bias",
            "readout_force.2.weight", "readout_force.2.bias",
        }
        bad_missing = {
            name for name in missing
            if name not in allowed_missing and ".vecnorm.dist_norm." not in name
        }
        if bad_missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={sorted(bad_missing)}, unexpected={unexpected}")

        # Frozen legacy head retained solely for existing checkpoint compatibility.
        head_input_dim = 512
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def set_trainable_backbone(self, last_n_layers: int) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if last_n_layers <= 0:
            return
        for block in self.backbone.Interaction[-last_n_layers:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        if last_n_layers >= len(self.backbone.Interaction):
            for parameter in self.backbone.embedding.parameters():
                parameter.requires_grad = True
        for module in (self.backbone.out_norm, self.backbone.edge_embedding, self.backbone.radial_fn):
            for parameter in module.parameters():
                parameter.requires_grad = True

    @staticmethod
    def _encode(backbone, z, pos, atom_batch, edge_index, n_graphs):
        node_scalar = backbone.embedding(z)
        edge_index, dist, edge_vector = backbone.connect(pos, edge_index, None)
        edge_feats = backbone.edge_embedding(backbone.radial_fn(dist))
        node_vector = torch.zeros(
            node_scalar.shape[0], 3, node_scalar.shape[1],
            dtype=node_scalar.dtype, device=node_scalar.device,
        )
        for interaction in backbone.Interaction:
            node_scalar, node_vector, edge_feats = interaction(
                node_scalar=node_scalar,
                node_vector=node_vector,
                edge_index=edge_index,
                dist=dist,
                edge_feats=edge_feats,
                edge_vector=edge_vector,
            )
        node_scalar = backbone.out_norm(node_scalar)
        summed = _pool_sum(node_scalar, atom_batch, n_graphs)
        counts = torch.bincount(atom_batch, minlength=n_graphs).to(node_scalar.dtype).clamp_min(1)
        mean = summed / counts[:, None]
        scaled_sum = summed / counts.sqrt()[:, None]
        return torch.cat([mean, scaled_sum], dim=-1)
