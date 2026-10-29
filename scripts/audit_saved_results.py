"""Reconcile historical prediction/metric exports without changing source records.

This is a result-consistency audit, NOT a certificate of training/test isolation.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ncmol.protocol import molecule_id
from ncmol.utils import metrics


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def audit(predictions_path, metrics_path):
    predictions, reported = read(predictions_path), read(metrics_path)
    fields = ("section", "family", "model", "endpoint", "split", "partition")
    key = lambda row: tuple(row[field] for field in fields)
    groups, metric_groups = defaultdict(list), defaultdict(list)
    for row in predictions:
        groups[key(row)].append(row)
    for row in reported:
        metric_groups[key(row)].append(row)
    issues, checks, comparisons = [], [], defaultdict(list)
    for group, rows in sorted(groups.items()):
        label = dict(zip(fields, group))
        ids = [molecule_id(row["smiles"]) for row in rows]
        if len(set(ids)) != len(ids):
            issues.append({**label, "issue": "duplicate molecules", "count": len(ids)-len(set(ids))})
        truth = [float(row["y_true"]) for row in rows]
        score = metrics(truth, [float(row["y_pred"]) for row in rows])
        references = metric_groups[group]
        if len(references) != 1:
            issues.append({**label, "issue": "metric row missing or duplicated", "count": len(references)})
        deltas = {}
        for ref in references:
            for name, value in score.items():
                if ref.get(name) in (None, "", "nan", "NaN"):
                    continue
                delta = abs(float(ref[name]) - value)
                deltas[name] = delta
                if not math.isfinite(delta) or delta > 1e-5:
                    issues.append({**label, "issue": "metric mismatch", "metric": name,
                                   "reported": float(ref[name]), "recomputed": value})
            if ref.get("n_eval") and int(float(ref["n_eval"])) != len(rows):
                issues.append({**label, "issue": "n_eval mismatch"})
        checks.append({**label, "n": len(rows),
                       "metrics": {k: v if math.isfinite(v) else None for k, v in score.items()},
                       "absolute_differences": deltas})
        comparisons[(group[3], group[4], group[5])].append((label, dict(zip(ids, truth))))
    for group, members in comparisons.items():
        reference_label, reference = members[0]
        for label, values in members[1:]:
            if values.keys() != reference.keys():
                issues.append({**label, "issue": "different evaluation membership",
                               "reference_model": reference_label["model"]})
            elif any(abs(values[i] - reference[i]) > 1e-6 for i in values):
                issues.append({**label, "issue": "different ground-truth labels"})
    for group in metric_groups:
        if group[-1] == "test" and group not in groups:
            issues.append({**dict(zip(fields, group)), "issue": "test predictions missing"})
    return {
        "scope": "saved predictions and reported metrics only",
        "training_leakage_status": "UNVERIFIED: original splits, checkpoints and baseline pipelines required",
        "sources": [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in (predictions_path, metrics_path)],
        "prediction_rows": len(predictions), "evaluation_groups": len(groups),
        "matched_evaluation_sets": len(comparisons), "issues": issues, "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() in (args.predictions.resolve(), args.metrics.resolve()):
        parser.error("Use a new output path; existing evidence must not be overwritten")
    report = audit(args.predictions, args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: report[name] for name in
                     ("prediction_rows", "evaluation_groups", "matched_evaluation_sets", "issues")}))
    return bool(report["issues"])


if __name__ == "__main__":
    sys.exit(main())
