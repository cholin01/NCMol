"""Strictly load every checkpoint under selected run directories."""
import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ncmol.inference import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = sorted({p for directory in args.directories for p in directory.rglob("best.pt")})
    if not paths:
        raise ValueError("No best.pt files found")
    for i, path in enumerate(paths, 1):
        model, mean, std = load_model(path)
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError(f"Non-finite model parameters: {path}")
        print(f"PASS {i}/{len(paths)} {path}: JEPA={model.jepa_predictor is not None}, mean={mean:.6g}, std={std:.6g}", flush=True)
        del model
        gc.collect()
    print(f"ALL {len(paths)} CHECKPOINTS LOADED STRICTLY", flush=True)


if __name__ == "__main__":
    main()
