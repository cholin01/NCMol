"""Predict from a SMILES CSV or an existing explicit-H processed dataset."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from ncmol.inference import graph_from_smiles, load_model
from ncmol.utils import move_batch
from train import collate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trusted trained NCMol best.pt")
    parser.add_argument("--vendor-dir", type=Path, default=Path("vendor/LiTEN-FF"))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-csv", type=Path)
    inputs.add_argument("--data", type=Path, help="Trusted processed explicit-H dataset .pt")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--strategy", choices=["random", "cliff", "scaffold"], default="random")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--candidate-conformers", type=int, default=10)
    parser.add_argument("--keep-conformers", type=int, default=3)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    if args.output.resolve() in {p.resolve() for p in (args.input_csv, args.data, args.checkpoint) if p}:
        parser.error("Output must not overwrite an input/checkpoint")
    torch.set_num_threads(args.threads)
    model, mean, std = load_model(args.checkpoint, args.vendor_dir, args.device)
    if args.data:
        payload = torch.load(args.data, map_location="cpu", weights_only=False)
        records = payload["splits"][args.strategy][args.split]
    else:
        with args.input_csv.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if args.smiles_column not in (reader.fieldnames or []):
                parser.error(f"CSV must include {args.smiles_column!r}")
            records = list(reader)
    if not records:
        parser.error("Input contains no molecules")
    results = []
    for start in range(0, len(records), args.batch_size):
        items = []
        for offset, record in enumerate(records[start:start + args.batch_size]):
            row = dict(record)
            smiles = row["smiles"] if args.data else row[args.smiles_column]
            try:
                graph = payload["graphs"][row["key"]] if args.data else graph_from_smiles(
                    smiles, args.candidate_conformers, args.keep_conformers)
                if not graph["conformers"]:
                    raise ValueError("No conformers")
                for conf in graph["conformers"]:
                    if not torch.isfinite(conf["pos"]).all():
                        raise ValueError("Non-finite coordinates")
                items.append((graph, {"smiles": smiles, "y": float(row.get("y", 0)),
                                      "input_index": start + offset}))
            except Exception as exc:
                raise ValueError(f"Input molecule {start + offset}: {exc}") from exc
        batch = move_batch(collate(items), torch.device(args.device))
        with torch.inference_mode():
            predictions = model.molecule_predictions(batch) * std + mean
        if not torch.isfinite(predictions).all():
            raise RuntimeError("Non-finite predictions; output was not written")
        for (graph, row), prediction in zip(items, predictions.cpu().tolist(), strict=True):
            result = {"input_index": row["input_index"], "smiles": row["smiles"],
                      "y_pred": prediction, "n_conformers": len(graph["conformers"])}
            if args.data:
                result["y_true"] = row["y"]
            results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Predicted {len(results)} molecules on the original modelling scale: {args.output}")


if __name__ == "__main__":
    main()
