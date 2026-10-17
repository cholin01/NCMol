"""Verify a trained checkpoint against its saved test predictions."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ncmol.data import collate
from ncmol.inference import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--strategy", default="random")
    parser.add_argument("--limit", type=int, default=8, help="0 checks all test molecules")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, mean, std = load_model(args.checkpoint)
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    rows = payload["splits"][args.strategy]["test"]
    with args.reference.open() as handle:
        reference = list(csv.DictReader(handle))
    assert len(reference) == len(rows), "Reference/test set size mismatch"
    indices = np.linspace(0, len(rows) - 1, min(args.limit, len(rows)), dtype=int) if args.limit else range(len(rows))
    errors = []
    for index in indices:
        row, expected = rows[index], reference[index]
        assert row["smiles"] == expected["smiles"], "Reference row ordering mismatch"
        batch = collate([(payload["graphs"][row["key"]], row)])
        with torch.inference_mode():
            prediction = float((model.molecule_predictions(batch) * std + mean)[0])
        error = abs(prediction - float(expected["y_pred"]))
        errors.append(error)
        assert np.isfinite(prediction) and error < 1e-4, (index, prediction, expected)
    print(f"PASS: {len(errors)} real checkpoint predictions; max absolute difference={max(errors):.9g}")


if __name__ == "__main__":
    main()
