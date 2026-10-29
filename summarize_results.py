from __future__ import annotations

# Report every available run in a fixed order. Never rank, filter, or select
# model variants using held-out test metrics.

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def add_result(rows, stage, endpoint, split, variant, directory):
    metrics_path = directory / "metrics.json"
    if not metrics_path.exists():
        return
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    test = result["test"]
    val = result["val"]
    rows.append({
        "stage": stage,
        "endpoint": endpoint,
        "split": split,
        "variant": variant,
        "best_epoch": result["best_epoch"],
        "val_rmse": val["rmse"],
        "test_rmse": test["rmse"],
        "test_mae": test["mae"],
        "test_r2": test["r2"],
        "test_pearson": test["pearson"],
        "test_spearman": test["spearman"],
        "test_ci": test["ci"],
    })


def main():
    rows = []
    for split in ("random", "cliff", "scaffold"):
        add_result(
            rows, "pampa", "pampa", split, "scratch",
            ROOT / "runs/ncmol_pampa" / split,
        )
        add_result(
            rows, "pampa", "pampa", split, "scratch_jepa",
            ROOT / "runs/ncmol_pampa_jepa" / split,
        )
    endpoints = (
        "caco2_log_papp_ab", "caco2_log_papp_ba", "caco2_log10_er",
        "mdck_log_papp_ab", "mdck_log10_er",
    )
    for endpoint in endpoints:
        for split in ("random", "scaffold"):
            for directory_variant, variant in (("base", "scratch"), ("jepa", "scratch_jepa")):
                add_result(
                    rows, "direct_cell", endpoint, split, variant,
                    ROOT / "runs/ncmol_cell" / endpoint / split / directory_variant,
                )
            for directory_variant, variant in (("base", "transfer"), ("jepa", "transfer_jepa")):
                add_result(
                    rows, "pampa_transfer", endpoint, split, variant,
                    ROOT / "runs/ncmol_pampa_transfer" / endpoint / split / directory_variant,
                )
    output = ROOT / "runs/ncmol_all_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["stage", "endpoint", "split", "variant"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"FINAL {output} rows={len(rows)}")


if __name__ == "__main__":
    main()
