"""Read-only audit of the original experiment tree; writes a separate JSON report.

Run in the original experiment environment (RDKit, torch, pandas, sklearn,
xgboost). No training, checkpoint modification, or split repair is performed.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


@lru_cache(None)
def identity(smiles, policy="exact"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES in original data")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    mol = Chem.RemoveHs(mol)
    if policy == "parent_tautomer":
        mol = rdMolStandardize.ChargeParent(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    if policy == "scaffold":
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or "<acyclic>"
    return Chem.MolToSmiles(mol, isomericSmiles=policy != "connectivity")


def partition_report(parts):
    report = {"sizes": {p: len(rows) for p, rows in parts.items()}, "policies": {}}
    for policy in ("exact", "connectivity", "parent_tautomer", "scaffold"):
        ids = {p: [identity(r["smiles"], policy) for r in rows] for p, rows in parts.items()}
        report["policies"][policy] = {
            "within_partition_duplicates": {p: len(v)-len(set(v)) for p, v in ids.items()},
            "cross_partition_overlap": {a+"/"+b: len(set(ids[a]) & set(ids[b]))
                                        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
        }
    return report


def row_map(rows):
    return {identity(r["smiles"]): float(r.get("y", r.get("y_true"))) for r in rows}


def compare_rows(left, right):
    a, b = row_map(left), row_map(right)
    return {"same_members": a.keys() == b.keys(),
            "left_only": len(a.keys()-b.keys()), "right_only": len(b.keys()-a.keys()),
            "max_shared_label_difference": max((abs(a[k]-b[k]) for k in a.keys() & b.keys()), default=0)}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file; do not overwrite audit evidence")
    root = args.root.resolve()
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / "src"))
    from run_cell_assay_baselines import ENDPOINTS, load_endpoint, fixed_splits
    report = {"rdkit": rdBase.rdkitVersion, "data": {}, "baseline_partitions": {},
              "runs": [], "transfer": [], "source_hashes": {}}
    paths = [root / "data/processed/npmmpd_2d3d_explicit_h.pt"] + sorted(
        (root / "data/processed/cell_2d3d_explicit_h").glob("*.pt"))
    data = {}
    for path in paths:
        print("AUDIT DATA " + path.name, file=sys.stderr, flush=True)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        splits = payload["splits"]
        mismatches = sum(identity(row["smiles"]) != identity(payload["graphs"][row["key"]]["smiles"])
                         for parts in splits.values() for rows in parts.values() for row in rows)
        data[str(path.relative_to(root))] = splits
        report["data"][str(path.relative_to(root))] = {
            "sha256": digest(path), "graph_row_mismatches": mismatches,
            "splits": {s: partition_report(parts) for s, parts in splits.items()},
        }
        del payload
    graph_source = torch.load(root / "data/processed/cell_assay_graphs.pt", map_location="cpu", weights_only=False)
    graph_keys = set(graph_source["graphs"])
    del graph_source
    for endpoint, (filename, transform) in ENDPOINTS.items():
        frame, metadata = load_endpoint(filename, transform)
        filtered = frame[frame["smiles"].isin(graph_keys)].reset_index(drop=True)
        checks = {"cleaning": metadata, "conformer_filter_removed": len(frame)-len(filtered), "splits": {}}
        for strategy, indices in fixed_splits(frame):
            graph_indices = dict(fixed_splits(filtered))[strategy]
            target = data[f"data/processed/cell_2d3d_explicit_h/{endpoint}.pt"][strategy]
            checks["splits"][strategy] = {}
            for partition, index in indices.items():
                traditional = [{"smiles": frame.iloc[i]["smiles"], "y": frame.iloc[i]["target"]} for i in index]
                graph = [{"smiles": filtered.iloc[i]["smiles"], "y": filtered.iloc[i]["target"]}
                         for i in graph_indices[partition]]
                checks["splits"][strategy][partition] = {
                    "traditional_vs_ncmol": compare_rows(traditional, target[partition]),
                    "graph_vs_ncmol": compare_rows(graph, target[partition]),
                }
        report["baseline_partitions"][endpoint] = checks
    patterns = ["runs/pampa_recurrent_2d_conditioned_scratch30/*/best.pt",
                "runs/latest_recurrent_scratch30_jepa/pampa/*/best.pt",
                "runs/latest_model_cells/*/*/*/best.pt", "runs/latest_model_pampa_transfer/*/*/*/best.pt",
                "runs/cell_graph_baselines/*/*/*/best.pt"]
    run_records = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            name = str(path.parent.relative_to(root))
            print("AUDIT RUN " + name, file=sys.stderr, flush=True)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            config = json.loads((path.parent / "config.json").read_text())
            graph_run = "cell_graph_baselines" in name
            strategy = config.get("strategy", config.get("split"))
            dataset = config.get("data", f"data/processed/cell_2d3d_explicit_h/{config.get('endpoint')}.pt")
            dataset = str(Path(dataset).relative_to(root)) if Path(dataset).is_absolute() else dataset
            parts = data[dataset][strategy]
            train_y = np.asarray([r["y"] for r in parts["train"]], dtype=np.float32)
            history = [json.loads(line) for line in (path.parent / "history.jsonl").read_text().splitlines() if line.strip()]
            best_rmse, best_epoch = math.inf, 0
            tolerance = 1e-6 if graph_run else 1e-5
            for record in history:
                if record["val"]["rmse"] < best_rmse - tolerance:
                    best_rmse, best_epoch = record["val"]["rmse"], record["epoch"]
            result = {"run": name, "dataset": dataset, "strategy": strategy,
                      "epochs_logged": len(history), "configured_epochs": config["epochs"],
                      "test_logged_epochs": sum("test" in r for r in history),
                      "checkpoint_epoch": ckpt["epoch"], "validation_rule_epoch": best_epoch,
                      "validation_selection_matches": ckpt["epoch"] == best_epoch,
                      "target_mean_difference": abs(float(ckpt["target_mean"])-float(train_y.mean())),
                      "target_std_difference": abs(float(ckpt["target_std"])-max(float(train_y.std()), 1e-8)),
                      "initialization": str(config.get("checkpoint")), "model_init": config.get("model_init"),
                      "checkpoint_sha256": digest(path), "history_sha256": digest(path.parent / "history.jsonl"),
                      "config_sha256": digest(path.parent / "config.json"), "predictions": {}}
            for partition in ("val", "test"):
                predpath = path.parent / f"predictions_{partition}.csv"
                result["predictions"][partition] = compare_rows(read_csv(predpath), parts[partition])
            report["runs"].append(result)
            run_records[name] = (parts, config)
            del ckpt
    for name, (parts, config) in run_records.items():
        source = config.get("model_init")
        if not source:
            continue
        source = Path(source)
        source_name = str(source.parent.relative_to(root)) if source.is_absolute() else str(source.parent)
        if source_name not in run_records:
            report["transfer"].append({"run": name, "error": "missing source run", "source": source_name})
            continue
        source_parts = run_records[source_name][0]
        for policy in ("exact", "connectivity", "parent_tautomer"):
            overlap = {}
            for sp in ("train", "val", "test"):
                for tp in ("train", "val", "test"):
                    a = {identity(r["smiles"], policy) for r in source_parts[sp]}
                    b = {identity(r["smiles"], policy) for r in parts[tp]}
                    common = a & b
                    overlap[sp+"/"+tp] = {"count": len(common), "molecule_hashes": sorted(
                        hashlib.sha256(s.encode()).hexdigest() for s in common)}
            report["transfer"].append({"run": name, "source": source_name, "policy": policy, "overlap": overlap})
    for path in sorted((root / "src").glob("*.py")):
        report["source_hashes"][str(path.relative_to(root))] = digest(path)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "datasets": len(data), "runs": len(report["runs"])}))


if __name__ == "__main__":
    main()
