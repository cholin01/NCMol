"""Build explicit-H 2D/3D data from a labelled CSV with user-specified splits."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from rdkit import Chem

from ncmol.inference import graph_from_smiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV columns: smiles,y,split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=["random", "cliff", "scaffold"], default="random")
    parser.add_argument("--candidate-conformers", type=int, default=10)
    parser.add_argument("--keep-conformers", type=int, default=3)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must differ from input")
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"smiles", "y", "split"} <= set(reader.fieldnames or []):
            parser.error("CSV must contain smiles,y,split")
        rows = list(reader)
    graphs, splits, seen = {}, {name: [] for name in ("train", "val", "test")}, set()
    for index, row in enumerate(rows):
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid SMILES in row {index + 2}")
        key = Chem.MolToSmiles(mol, isomericSmiles=True)
        if key in seen:
            raise ValueError(f"Duplicate molecule in row {index + 2}; aggregate measurements before splitting")
        seen.add(key)
        y = float(row["y"])
        if not math.isfinite(y) or row["split"] not in splits:
            raise ValueError(f"Invalid label/split in row {index + 2}")
        graphs[key] = graph_from_smiles(row["smiles"], args.candidate_conformers, args.keep_conformers)
        splits[row["split"]].append({"smiles": row["smiles"], "key": key, "y": y})
    if not all(splits.values()):
        raise ValueError("Specify nonempty train, val and test sets")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"graphs": graphs, "splits": {args.strategy: splits}, "metadata": {
        "conformers": "ETKDGv3 + MMFF94s/UFF, energy-ranked", "cutoff": 5.0,
        "max_neighbors": 32, "candidate_conformers": args.candidate_conformers,
        "keep_conformers": args.keep_conformers, "split_source": "user-specified CSV",
    }}, args.output)
    print(f"Saved {len(graphs)} molecules: {args.output}")


if __name__ == "__main__":
    main()
