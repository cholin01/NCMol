from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ncmol.model import NCMol
from ncmol.data import TwoDThreeDDataset, collate
from ncmol.utils import metrics, move_batch, seed_everything
from ncmol.protocol import audit_splits, initialization_exposure


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
    parser.add_argument("--task-id", default="unspecified", help="Assay endpoint identifier, e.g. pampa")
    parser.add_argument("--transfer-policy", choices=["molecule-disjoint", "cross-task"], default="molecule-disjoint")
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
    identities = audit_splits(payload, args.strategy)
    exposed = set(identities["train"]) | set(identities["val"])
    by_task = {args.task_id: set(exposed)}
    if args.transfer_policy == "cross-task" and args.task_id == "unspecified":
        parser.error("--transfer-policy cross-task requires --task-id")
    for source in (args.checkpoint, args.model_init):
        if source is not None:
            source_payload = torch.load(source, map_location="cpu", weights_only=False)
            exposed.update(initialization_exposure(
                source_payload, identities, args.transfer_policy, args.task_id,
            ))
            source_manifest = source_payload["protocol"]
            for task, ids in source_manifest.get("exposure_by_task", {
                "unspecified": source_manifest["selection_exposure_ids"],
            }).items():
                by_task.setdefault(task, set()).update(ids)
            del source_payload
    protocol = {"version": 2, "split_ids": identities,
                "selection_exposure_ids": sorted(exposed),
                "exposure_by_task": {task: sorted(ids) for task, ids in by_task.items()},
                "transfer_policy": args.transfer_policy,
                "test_policy": "once after validation-only checkpoint selection"}
    datasets = {
        split: TwoDThreeDDataset(payload, args.strategy, split)
        for split in ("train", "val")
    }
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
        args.checkpoint, args.dropout,
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
        scheduler.step(val_metrics["rmse"])
        record = {
            "epoch": epoch,
            "train_mse_normalized": total_supervised / max(total_items, 1),
            "train_alignment_loss": total_alignment / max(len(datasets["train"]), 1),
            "train_jepa_loss": total_jepa / max(total_items, 1),
            "val": val_metrics,
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
                "protocol": protocol,
            }, args.output_dir / "best.pt")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    # No test loader or test evaluation exists until checkpoint selection is finished.
    loaders["test"] = DataLoader(
        TwoDThreeDDataset(payload, args.strategy, "test"), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, collate_fn=collate,
    )
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
