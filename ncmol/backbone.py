from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .compat import install_fallbacks


def _pool_sum(x: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = x.new_zeros((size, x.shape[-1]))
    out.index_add_(0, index, x)
    return out


class SpatialEncoderRegressor(nn.Module):
    """3D equivariant encoder adapter used by NCMol."""

    def __init__(
        self, vendor_dir: str | Path, checkpoint: str | Path | None = None,
        dropout: float = 0.2, conformer_supervision: bool = False,
        jepa: bool = False, jepa_hidden_dim: int = 256,
    ):
        super().__init__()
        self.conformer_supervision = conformer_supervision
        install_fallbacks()
        vendor_dir = str(Path(vendor_dir).resolve())
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)
        from model.LiTEN_FF import LiTEN  # type: ignore

        self.backbone = LiTEN()
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

        # Mean and size-normalized sum pool provide complementary intensive and
        # size-sensitive summaries. Ensemble mean/std then aggregate conformers.
        head_input_dim = 512 if conformer_supervision else 1024
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
        self.jepa_predictor = None
        self.target_backbone = None
        if jepa:
            self.jepa_predictor = nn.Sequential(
                nn.LayerNorm(512),
                nn.Linear(512, jepa_hidden_dim),
                nn.GELU(),
                nn.Linear(jepa_hidden_dim, 512),
            )
            self.target_backbone = copy.deepcopy(self.backbone)
            self.target_backbone.requires_grad_(False)
            self.target_backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.target_backbone is not None:
            self.target_backbone.eval()
        return self

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

    def encode(self, z, pos, atom_batch, edge_index, n_graphs):
        return self._encode(
            self.backbone, z, pos, atom_batch, edge_index, n_graphs,
        )

    @torch.no_grad()
    def encode_target(self, batch):
        if self.target_backbone is None:
            raise RuntimeError("JEPA target encoder is not enabled")
        return self._encode(
            self.target_backbone, batch["z"], batch["pos"],
            batch["atom_batch"], batch["edge_index"], batch["n_graphs"],
        )

    def jepa_loss(self, batch, context_features):
        if self.jepa_predictor is None:
            raise RuntimeError("JEPA predictor is not enabled")
        target_features = self.encode_target(batch)
        paired_targets = torch.empty_like(target_features)
        molecule_index = batch["molecule_index"]
        for molecule in range(batch["n_molecules"]):
            indices = torch.nonzero(
                molecule_index == molecule, as_tuple=False,
            ).flatten()
            if len(indices) > 1:
                paired_targets[indices] = target_features[indices.roll(-1)]
            else:
                paired_targets[indices] = target_features[indices]
        prediction = self.jepa_predictor(context_features)
        prediction = F.normalize(prediction, dim=-1)
        paired_targets = F.normalize(paired_targets.detach(), dim=-1)
        return (prediction - paired_targets).square().sum(dim=-1).mean()

    @torch.no_grad()
    def update_jepa_target(self, momentum: float) -> None:
        if self.target_backbone is None:
            return
        for target, online in zip(
            self.target_backbone.parameters(), self.backbone.parameters(), strict=True,
        ):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)
        for target, online in zip(
            self.target_backbone.buffers(), self.backbone.buffers(), strict=True,
        ):
            target.copy_(online)

    def forward(
        self, batch, return_conformers: bool = False,
        return_features: bool = False,
    ):
        conformer_features = self.encode(
            batch["z"], batch["pos"], batch["atom_batch"],
            batch["edge_index"], batch["n_graphs"],
        )
        molecule_index = batch["molecule_index"]
        n_molecules = batch["n_molecules"]
        if self.conformer_supervision:
            conformer_predictions = self.head(conformer_features).squeeze(-1)
            if return_conformers:
                if return_features:
                    return conformer_predictions, conformer_features
                return conformer_predictions
            summed_predictions = _pool_sum(
                conformer_predictions[:, None], molecule_index, n_molecules,
            ).squeeze(-1)
            counts = torch.bincount(
                molecule_index, minlength=n_molecules,
            ).to(conformer_predictions.dtype).clamp_min(1)
            return summed_predictions / counts
        summed = _pool_sum(conformer_features, molecule_index, n_molecules)
        counts = torch.bincount(molecule_index, minlength=n_molecules).to(conformer_features.dtype).clamp_min(1)
        mean = summed / counts[:, None]
        # Keep ensemble dispersion as an input feature, but do not backpropagate
        # through the subtractive E[x^2] - E[x]^2 calculation. On nearly
        # identical conformers this path is ill-conditioned and can inject NaN
        # gradients into an otherwise stable pretrained backbone.
        detached = conformer_features.detach()
        detached_mean = _pool_sum(detached, molecule_index, n_molecules) / counts[:, None]
        squared = _pool_sum(detached.square(), molecule_index, n_molecules) / counts[:, None]
        # A positive floor is essential once the backbone is unfrozen: the
        # derivative of sqrt(var) is singular at exactly zero variance.
        std = (squared - detached_mean.square()).clamp_min(1e-8).sqrt()
        features = torch.cat([mean, std], dim=-1)
        return self.head(features).squeeze(-1)
