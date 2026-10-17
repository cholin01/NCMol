from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ncmol.backbone import SpatialEncoderRegressor
from ncmol.utils import metrics, move_batch, seed_everything


class TwoDThreeDDataset(Dataset):
    def __init__(self, payload, strategy: str, split: str):
        self.graphs = payload["graphs"]
        self.rows = payload["splits"][strategy][split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.graphs[row["key"]], row


def collate(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    zs, positions, edges, atom_batches = [], [], [], []
    molecule_indices, ys, rows = [], [], []
    z2d, x2d, edges2d, attrs2d, batches2d = [], [], [], [], []
    offset3d = 0
    offset2d = 0
    graph_index = 0
    aligned_3d_indices, aligned_to_2d_indices = [], []
    for molecule_index, (graph, row) in enumerate(items):
        n2d = len(graph["z_2d"])
        if not n2d or not graph["conformers"]:
            raise ValueError("Each molecule requires atoms and at least one conformer")
        for conformer in graph["conformers"]:
            z = conformer["z"]
            if len(z) != n2d or not torch.equal(z, graph["z_2d"]):
                raise RuntimeError(
                    "Explicit-H 2D/3D atom ordering mismatch; interaction is unsafe"
                )
            zs.append(z)
            positions.append(conformer["pos"])
            edges.append(conformer["edge_index"] + offset3d)
            atom_batches.append(torch.full((len(z),), graph_index, dtype=torch.long))
            molecule_indices.append(molecule_index)
            aligned_3d_indices.append(torch.arange(offset3d, offset3d + n2d))
            aligned_to_2d_indices.append(torch.arange(offset2d, offset2d + n2d))
            offset3d += len(z)
            graph_index += 1
        z2d.append(graph["z_2d"])
        x2d.append(graph["x_2d"])
        edges2d.append(graph["edge_index_2d"] + offset2d)
        attrs2d.append(graph["edge_attr_2d"])
        batches2d.append(torch.full((n2d,), molecule_index, dtype=torch.long))
        offset2d += n2d
        ys.append(row["y"])
        rows.append(row)
    return {
        "z": torch.cat(zs), "pos": torch.cat(positions),
        "edge_index": torch.cat(edges, dim=1),
        "atom_batch": torch.cat(atom_batches),
        "molecule_index": torch.tensor(molecule_indices, dtype=torch.long),
        "z_2d": torch.cat(z2d), "x_2d": torch.cat(x2d),
        "edge_index_2d": torch.cat(edges2d, dim=1),
        "edge_attr_2d": torch.cat(attrs2d), "batch_2d": torch.cat(batches2d),
        "aligned_3d_index": torch.cat(aligned_3d_indices),
        "aligned_to_2d_index": torch.cat(aligned_to_2d_indices),
        "y": torch.tensor(ys, dtype=torch.float32), "rows": rows,
        "n_graphs": graph_index, "n_molecules": len(items),
    }


def pool_mean(values, index, size):
    output = values.new_zeros((size, values.shape[-1]))
    output.index_add_(0, index, values)
    counts = torch.bincount(index, minlength=size).to(values.dtype).clamp_min(1)
    return output / counts[:, None]


def pool_sum(values, index, size):
    output = values.new_zeros((size, values.shape[-1]))
    output.index_add_(0, index, values)
    return output


class GatedPairMessageBlock(nn.Module):
    """Directed pair-message aggregation followed by recurrent node updates.

    This is intentionally not a LiTEN/TQA update.  A message and a feature-wise
    gate are produced jointly from receiver, sender, their difference, and the
    current bond state.  Messages are degree-normalized and a GRU decides how
    much of the aggregate replaces the previous atom state.  Bond states are
    then updated from both new endpoint states.
    """

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        pair_dim = hidden_dim * 4
        self.message_net = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim * 2), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.aggregate_norm = nn.LayerNorm(hidden_dim)
        self.node_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.edge_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        # Small learned residual scales make full-backbone fine-tuning stable.
        self.node_scale = nn.Parameter(torch.full((hidden_dim,), 1e-1))
        self.edge_scale = nn.Parameter(torch.full((hidden_dim,), 1e-1))
        self.ffn_scale = nn.Parameter(torch.full((hidden_dim,), 1e-1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, edges, edge_index):
        receiver, sender = edge_index
        normalized = self.node_norm(nodes)
        normalized_edges = self.edge_norm(edges)
        pair = torch.cat([
            normalized[receiver], normalized[sender],
            normalized[sender] - normalized[receiver], normalized_edges,
        ], dim=-1)
        raw_message, raw_gate = self.message_net(pair).chunk(2, dim=-1)
        messages = F.silu(raw_message) * torch.sigmoid(raw_gate)
        aggregated = nodes.new_zeros(nodes.shape)
        aggregated.index_add_(0, receiver, messages)
        degree = torch.bincount(receiver, minlength=len(nodes)).to(nodes.dtype)
        aggregated = aggregated / degree.clamp_min(1).sqrt()[:, None]
        proposal = self.node_update(self.aggregate_norm(aggregated), nodes)
        nodes = nodes + self.node_scale * self.dropout(proposal - nodes)

        edge_delta = self.edge_update(torch.cat([
            self.node_norm(nodes)[receiver], self.node_norm(nodes)[sender],
            (nodes[sender] - nodes[receiver]).abs(), normalized_edges,
        ], dim=-1))
        edges = edges + self.edge_scale * self.dropout(edge_delta)
        nodes = nodes + self.ffn_scale * self.dropout(self.ffn(self.ffn_norm(nodes)))
        return nodes, edges


class RecurrentTopologyEncoder(nn.Module):
    """Explicit-H bond graph encoder with gated recurrent message passing."""

    def __init__(
        self, feature_dim: int = 24, edge_dim: int = 6,
        hidden_dim: int = 256, num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Element embeddings are shared with the 3D backbone,
        # so both modalities start from the same atom identity representation.
        self.node_input = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.edge_input = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GatedPairMessageBlock(hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def initialize(self, batch, shared_atom_embedding):
        nodes = self.node_input(torch.cat([
            shared_atom_embedding, batch["x_2d"],
        ], dim=-1))
        edges = self.edge_input(batch["edge_attr_2d"])
        return nodes, edges


class TopologyConditionedSpatialUpdate(nn.Module):
    """One-way topology-to-conformer conditioning at aligned atoms."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.norm_2d = nn.LayerNorm(hidden_dim)
        self.norm_3d = nn.LayerNorm(hidden_dim)
        self.to_3d = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.gate_3d = nn.Linear(hidden_dim * 3, hidden_dim)
        self.scale = nn.Parameter(torch.full((hidden_dim,), 1e-2))

    def forward(self, h2, h3, aligned_3d_index, aligned_to_2d_index):
        n2 = self.norm_2d(h2)
        n3 = self.norm_3d(h3)
        aligned_3d = n3[aligned_3d_index]
        aligned_2d = n2[aligned_to_2d_index]
        joint = torch.cat([aligned_3d, aligned_2d, aligned_2d - aligned_3d], dim=-1)
        gate_3d = torch.sigmoid(self.gate_3d(joint))
        delta_3d = h3.new_zeros(h3.shape)
        delta_3d.index_copy_(
            0, aligned_3d_index, self.scale * gate_3d * self.to_3d(joint),
        )
        return h2, h3 + delta_3d


class NCMol(nn.Module):
    """Topology-conditioned multi-conformer permeability model."""
    def __init__(
        self, vendor_dir, checkpoint=None, dropout=0.2,
        jepa=False, jepa_hidden_dim=256,
    ):
        super().__init__()
        self.three_d = SpatialEncoderRegressor(
            vendor_dir, checkpoint, dropout=dropout, conformer_supervision=True,
        )
        self.three_d.head.requires_grad_(False)
        self.two_d = RecurrentTopologyEncoder(dropout=dropout * 0.5)
        # Condition the spatial stream only at middle/late depths.  The 2D
        # stream remains topology-specific instead of being overwritten by 3D.
        self.exchange_after = (2, 4)
        self.exchanges = nn.ModuleList([
            TopologyConditionedSpatialUpdate() for _ in self.exchange_after
        ])
        self.project_3d = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 256), nn.SiLU())
        self.project_2d = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 256), nn.SiLU())
        self.gate = nn.Linear(512, 256)
        self.predict_2d_to_3d = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256),
        )
        self.predict_3d_to_2d = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, 256), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.SiLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )
        self.jepa_predictor = None
        self.target_backbone = None
        self.target_project_3d = None
        if jepa:
            self.jepa_predictor = nn.Sequential(
                nn.LayerNorm(256), nn.Linear(256, jepa_hidden_dim), nn.GELU(),
                nn.Linear(jepa_hidden_dim, 256),
            )
            self.target_backbone = copy.deepcopy(self.three_d.backbone)
            self.target_project_3d = copy.deepcopy(self.project_3d)
            self.target_backbone.requires_grad_(False)
            self.target_project_3d.requires_grad_(False)
            self.target_backbone.eval()
            self.target_project_3d.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.target_backbone is not None:
            self.target_backbone.eval()
            self.target_project_3d.eval()
        return self

    def set_trainable_backbone(self):
        self.three_d.set_trainable_backbone(6)

    def load_encoder_checkpoint(self, checkpoint):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        # The permeability regression head is assay-specific.  All online
        # representation modules, fusion gates and JEPA predictor are adapted.
        encoder_state = {
            key: value for key, value in state.items()
            if not key.startswith("head.")
            and not key.startswith("three_d.head.")
            and not key.startswith("target_backbone.")
            and not key.startswith("target_project_3d.")
        }
        missing, unexpected = self.load_state_dict(encoder_state, strict=False)
        allowed = ("head.", "three_d.head.", "target_backbone.", "target_project_3d.",
                   "jepa_predictor.")
        bad_missing = [key for key in missing if not key.startswith(allowed)]
        bad_unexpected = [key for key in unexpected if not key.startswith("jepa_predictor.")]
        if bad_missing or bad_unexpected:
            raise ValueError(f"Incompatible encoder checkpoint: missing={bad_missing}, unexpected={bad_unexpected}")
        self.reset_jepa_target()

    @torch.no_grad()
    def reset_jepa_target(self):
        if self.target_backbone is None:
            return
        self.target_backbone.load_state_dict(self.three_d.backbone.state_dict())
        self.target_project_3d.load_state_dict(self.project_3d.state_dict())

    @torch.no_grad()
    def encode_jepa_target(self, batch):
        if self.target_backbone is None:
            raise RuntimeError("JEPA target encoder is not enabled")
        pooled = SpatialEncoderRegressor._encode(
            self.target_backbone, batch["z"], batch["pos"],
            batch["atom_batch"], batch["edge_index"], batch["n_graphs"],
        )
        return self.target_project_3d(pooled)

    def multiconformer_jepa_loss(self, batch, online_features, target_features):
        if self.jepa_predictor is None:
            raise RuntimeError("JEPA predictor is not enabled")
        paired_targets = torch.empty_like(target_features)
        molecule_index = batch["molecule_index"]
        for molecule in range(batch["n_molecules"]):
            indices = torch.nonzero(molecule_index == molecule, as_tuple=False).flatten()
            paired_targets[indices] = target_features[
                indices.roll(-1) if len(indices) > 1 else indices
            ]
        prediction = F.normalize(self.jepa_predictor(online_features), dim=-1)
        paired_targets = F.normalize(paired_targets.detach(), dim=-1)
        return (prediction - paired_targets).square().sum(dim=-1).mean()

    @torch.no_grad()
    def update_jepa_target(self, momentum):
        if self.target_backbone is None:
            return
        for target, online in zip(
            self.target_backbone.parameters(), self.three_d.backbone.parameters(), strict=True,
        ):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)
        for target, online in zip(
            self.target_project_3d.parameters(), self.project_3d.parameters(), strict=True,
        ):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)
        for target, online in zip(
            self.target_backbone.buffers(), self.three_d.backbone.buffers(), strict=True,
        ):
            target.copy_(online)
        for target, online in zip(
            self.target_project_3d.buffers(), self.project_3d.buffers(), strict=True,
        ):
            target.copy_(online)

    def forward(self, batch, return_views=False):
        backbone = self.three_d.backbone
        h3 = backbone.embedding(batch["z"])
        edge_index, distances, edge_vectors = backbone.connect(
            batch["pos"], batch["edge_index"], None,
        )
        edge_3d = backbone.edge_embedding(backbone.radial_fn(distances))
        v3 = torch.zeros(
            h3.shape[0], 3, h3.shape[1], dtype=h3.dtype, device=h3.device,
        )
        shared_2d_embedding = backbone.embedding(batch["z_2d"])
        h2, edge_2d = self.two_d.initialize(batch, shared_2d_embedding)
        exchange_index = 0
        for layer_index, (layer_3d, layer_2d) in enumerate(zip(
            backbone.Interaction, self.two_d.layers, strict=True,
        )):
            h3, v3, edge_3d = layer_3d(
                node_scalar=h3, node_vector=v3, edge_index=edge_index,
                dist=distances, edge_feats=edge_3d, edge_vector=edge_vectors,
            )
            h2, edge_2d = layer_2d(h2, edge_2d, batch["edge_index_2d"])
            if layer_index in self.exchange_after:
                h2, h3 = self.exchanges[exchange_index](
                    h2, h3, batch["aligned_3d_index"], batch["aligned_to_2d_index"],
                )
                exchange_index += 1
        h3 = backbone.out_norm(h3)
        h2 = self.two_d.out_norm(h2)

        summed_3d = pool_sum(h3, batch["atom_batch"], batch["n_graphs"])
        counts_3d = torch.bincount(
            batch["atom_batch"], minlength=batch["n_graphs"],
        ).to(h3.dtype).clamp_min(1)
        h3 = self.project_3d(torch.cat([
            summed_3d / counts_3d[:, None], summed_3d / counts_3d.sqrt()[:, None],
        ], dim=-1))

        summed_2d = pool_sum(h2, batch["batch_2d"], batch["n_molecules"])
        counts_2d = torch.bincount(
            batch["batch_2d"], minlength=batch["n_molecules"],
        ).to(h2.dtype).clamp_min(1)
        h2 = self.project_2d(torch.cat([
            summed_2d / counts_2d[:, None], summed_2d / counts_2d.sqrt()[:, None],
        ], dim=-1))
        h2_conformer = h2[batch["molecule_index"]]
        gate = torch.sigmoid(self.gate(torch.cat([h3, h2_conformer], dim=-1)))
        fused = gate * h3 + (1.0 - gate) * h2_conformer
        joint = torch.cat([fused, h3, h2_conformer, (h3 - h2_conformer).abs()], dim=-1)
        predictions = self.head(joint).squeeze(-1)
        if return_views:
            h3_molecule = pool_mean(
                h3, batch["molecule_index"], batch["n_molecules"],
            )
            return predictions, h2, h3_molecule, h3
        return predictions

    def molecule_predictions(self, batch):
        predictions = self(batch)
        return pool_mean(
            predictions[:, None], batch["molecule_index"], batch["n_molecules"],
        ).squeeze(-1)


    def cross_view_loss(self, h2, h3):
        prediction_3d = F.normalize(self.predict_2d_to_3d(h2), dim=-1)
        prediction_2d = F.normalize(self.predict_3d_to_2d(h3), dim=-1)
        target_3d = F.normalize(h3.detach(), dim=-1)
        target_2d = F.normalize(h2.detach(), dim=-1)
        return 0.5 * (
            (prediction_3d - target_3d).square().sum(dim=-1).mean()
            + (prediction_2d - target_2d).square().sum(dim=-1).mean()
        )


@torch.no_grad()
def evaluate(model, loader, device, target_mean, target_std):
    model.eval()
    predictions, targets, rows = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model.molecule_predictions(batch) * target_std + target_mean
        predictions.extend(pred.cpu().tolist())
        targets.extend(batch["y"].cpu().tolist())
        rows.extend(batch["rows"])
    return metrics(targets, predictions), np.asarray(targets), np.asarray(predictions), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/processed/npmmpd_2d3d_explicit_h.pt"),
    )
    parser.add_argument("--strategy", choices=["random", "cliff", "scaffold"], required=True)
    parser.add_argument("--vendor-dir", type=Path, default=Path("vendor/LiTEN-FF"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=None, help="Optional backbone initialization; omit for random initialization",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--jepa-weight", type=float, default=0.0)
    parser.add_argument("--jepa-momentum", type=float, default=0.996)
    parser.add_argument("--jepa-hidden-dim", type=int, default=256)
    parser.add_argument("--model-init", type=Path)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=1, help="CPU threads per process")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.threads < 1:
        parser.error("epochs/batch-size/threads must be positive; num-workers must be nonnegative")
    if not 0 <= args.jepa_momentum < 1 or args.jepa_weight < 0 or args.alignment_weight < 0:
        parser.error("Invalid JEPA momentum or loss weights")
    if (args.output_dir / "best.pt").exists() or (args.output_dir / "history.jsonl").exists():
        parser.error("Output already contains a run; use a new --output-dir")
    torch.set_num_threads(args.threads)

    seed_everything(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    datasets = {
        split: TwoDThreeDDataset(payload, args.strategy, split)
        for split in ("train", "val", "test")
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("Training, validation and test sets must all be nonempty")
    keys = {name: {row["key"] for row in dataset.rows} for name, dataset in datasets.items()}
    if keys["train"] & keys["val"] or keys["train"] & keys["test"] or keys["val"] & keys["test"]:
        raise ValueError("Molecule keys overlap between training/validation/test sets")
    loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=split == "train",
            num_workers=args.num_workers, collate_fn=collate,
            pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
        )
        for split, dataset in datasets.items()
    }
    train_targets = np.asarray([row["y"] for row in datasets["train"].rows], dtype=np.float32)
    if not np.isfinite(train_targets).all():
        raise ValueError("Training labels must be finite")
    target_mean = float(train_targets.mean())
    target_std = max(float(train_targets.std()), 1e-8)
    model = NCMol(
        args.vendor_dir, args.checkpoint, args.dropout,
        jepa=args.jepa_weight > 0, jepa_hidden_dim=args.jepa_hidden_dim,
    ).to(device)
    if args.model_init is not None:
        model.load_encoder_checkpoint(args.model_init)
    model.set_trainable_backbone()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.8,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8",
    )
    history = args.output_dir / "history.jsonl"
    best_rmse = math.inf
    best_epoch = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_supervised = total_alignment = total_jepa = 0.0
        total_items = skipped_batches = 0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            normalized = (batch["y"] - target_mean) / target_std
            target = normalized[batch["molecule_index"]]
            optimizer.zero_grad(set_to_none=True)
            target_features = (
                model.encode_jepa_target(batch) if args.jepa_weight > 0 else None
            )
            prediction, h2, h3, h3_conformers = model(batch, return_views=True)
            supervised = (prediction - target).square().mean()
            alignment = model.cross_view_loss(h2, h3)
            jepa = (
                model.multiconformer_jepa_loss(batch, h3_conformers, target_features)
                if target_features is not None else supervised.new_zeros(())
            )
            loss = (
                supervised + args.alignment_weight * alignment
                + args.jepa_weight * jepa
            )
            if not bool(torch.isfinite(loss)):
                skipped_batches += 1
                continue
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            if not bool(torch.isfinite(grad_norm)):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            model.update_jepa_target(args.jepa_momentum)
            n = len(prediction)
            total_supervised += float(supervised.detach()) * n
            total_alignment += float(alignment.detach()) * batch["n_molecules"]
            total_jepa += float(jepa.detach()) * n
            total_items += n
        if total_items == 0:
            raise RuntimeError(f"Epoch {epoch}: every training batch was non-finite; no checkpoint saved")
        val_metrics, _, _, _ = evaluate(model, loaders["val"], device, target_mean, target_std)
        test_metrics, _, _, _ = evaluate(model, loaders["test"], device, target_mean, target_std)
        scheduler.step(val_metrics["rmse"])
        record = {
            "epoch": epoch,
            "train_mse_normalized": total_supervised / max(total_items, 1),
            "train_alignment_loss": total_alignment / max(len(datasets["train"]), 1),
            "train_jepa_loss": total_jepa / max(total_items, 1),
            "val": val_metrics, "test": test_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "skipped_batches": skipped_batches,
            "elapsed_seconds": time.time() - started,
        }
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "target_mean": target_mean, "target_std": target_std,
                "val_metrics": val_metrics, "args": vars(args),
            }, args.output_dir / "best.pt")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    results = {"best_epoch": best_epoch, "best_val_rmse": best_rmse}
    for split in ("val", "test"):
        score, truth, prediction, rows = evaluate(
            model, loaders[split], device, target_mean, target_std,
        )
        results[split] = score
        with (args.output_dir / f"predictions_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["smiles", "y_true", "y_pred"])
            for row, observed, predicted in zip(rows, truth, prediction, strict=True):
                writer.writerow([row["smiles"], observed, predicted])
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )
    print("FINAL " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
