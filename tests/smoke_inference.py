"""Real-backbone integration checks; temporary toy weights are not scientific results."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncmol.inference import graph_from_smiles, load_model
from ncmol.utils import metrics
from ncmol.data import collate


def run(*args, success=True):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                            text=True, capture_output=True)
    if success and result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    if not success and result.returncode == 0:
        raise AssertionError("Expected invalid input to fail")
    return result


def predictions(path):
    with path.open() as stream:
        return np.array([float(row["y_pred"]) for row in csv.DictReader(stream)])


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="ncmol-smoke-") as directory:
        work = Path(directory)
        source = work / "labelled.csv"
        source.write_text("smiles,y,split\nCCO,-6,train\nCCN,-5,train\nCCC,-7,train\n"
                          "CCCO,-4,train\nCO,-5.5,val\nCN,-6.5,val\n"
                          "CCCC,-6.2,test\nCCOC,-4.8,test\n")
        data = work / "data.pt"
        run("prepare_csv.py", "--input", source, "--output", data,
            "--candidate-conformers", 3, "--keep-conformers", 2)
        augmented = work / "augmented.pt"
        run("prepare_data.py", "--input", data, "--output", augmented)
        original = torch.load(data, weights_only=False)
        rebuilt = torch.load(augmented, weights_only=False)
        for key in original["graphs"]:
            torch.testing.assert_close(original["graphs"][key]["x_2d"],
                                       rebuilt["graphs"][key]["x_2d"])
        print("PASS: CSV -> explicit-H conformers and disjoint splits", flush=True)
        checkpoints = {}
        transfer_source = work / "transfer.csv"
        transfer_source.write_text("smiles,y,split\nCCO,-6,train\nCCN,-5,train\nCCC,-7,train\n"
                                   "CCCO,-4,train\nCCCl,-5.5,val\nCCBr,-6.5,val\n"
                                   "CCF,-6.2,test\nCCS,-4.8,test\n")
        transfer_data = work / "transfer.pt"
        run("prepare_csv.py", "--input", transfer_source, "--output", transfer_data,
            "--candidate-conformers", 3, "--keep-conformers", 2)
        for variant in ("base", "jepa", "transfer"):
            output = work / variant
            options = ["--jepa-weight", "0.1"] if variant != "base" else []
            if variant == "transfer":
                options += ["--model-init", checkpoints["jepa"]]
            run_data = transfer_data if variant == "transfer" else data
            run("train.py", "--data", run_data, "--strategy", "random",
                "--output-dir", output, "--device", "cpu", "--epochs", 1,
                "--batch-size", 2, "--num-workers", 0, "--threads", 1, *options)
            checkpoint = output / "best.pt"
            checkpoints[variant] = checkpoint
            history = json.loads((output / "history.jsonl").read_text())
            assert history["skipped_batches"] == 0
            assert "test" not in history
            saved = torch.load(checkpoint, weights_only=False)
            assert saved["protocol"]["version"] == 2
            assert np.isfinite(history["train_mse_normalized"])
            for batch_size in (1, 2):
                predicted = work / f"{variant}_{batch_size}.csv"
                run("predict.py", "--checkpoint", checkpoint,
                    "--data", run_data, "--output", predicted, "--batch-size", batch_size)
                np.testing.assert_allclose(predictions(predicted),
                                           predictions(output / "predictions_test.csv"), atol=2e-5, rtol=1e-5)
            print(f"PASS: {variant} training/backward/checkpoint/CLI inference agrees with evaluation", flush=True)
        model, mean, std = load_model(checkpoints["jepa"])
        # Changing held-out labels must not change training or checkpoint selection.
        perturbed = torch.load(data, weights_only=False)
        for row in perturbed["splits"]["random"]["test"]:
            row["y"] += 1000.0
        changed_data = work / "changed_test_labels.pt"
        torch.save(perturbed, changed_data)
        changed_output = work / "changed_test_run"
        run("train.py", "--data", changed_data, "--strategy", "random",
            "--output-dir", changed_output, "--device", "cpu", "--epochs", 1,
            "--batch-size", 2, "--num-workers", 0, "--threads", 1)
        before = torch.load(checkpoints["base"], weights_only=False)
        after = torch.load(changed_output / "best.pt", weights_only=False)
        assert before["epoch"] == after["epoch"]
        assert before["val_metrics"] == after["val_metrics"]
        for key in before["model"]:
            torch.testing.assert_close(before["model"][key], after["model"][key], rtol=0, atol=0)
        print("PASS: changing test labels leaves training weights and selection exactly unchanged", flush=True)
        macrocycle = graph_from_smiles("C1CCCCCCCCCCC1", candidates=4, keep=3)
        assert len(macrocycle["conformers"]) >= 2
        batch = collate([(macrocycle, {"y": 0.0})])
        with torch.inference_mode():
            direct = model(batch).mean() * std + mean
            pooled = model.molecule_predictions(batch)[0] * std + mean
        torch.testing.assert_close(direct, pooled)
        assert abs(mean - (-5.5)) < 1e-6
        assert abs(std - np.std([-6, -5, -7, -4])) < 1e-6
        print("PASS: multi-conformer mean and train-only label inverse transform", flush=True)
        smiles_csv = work / "smiles.csv"
        smiles_csv.write_text("smiles\nC1CCCCCCCCCCC1\nCCO\n")
        raw_output = work / "raw.csv"
        run("predict.py", "--checkpoint", checkpoints["jepa"],
            "--input-csv", smiles_csv, "--output", raw_output,
            "--candidate-conformers", 4, "--keep-conformers", 3)
        assert len(predictions(raw_output)) == 2 and np.isfinite(predictions(raw_output)).all()
        print("PASS: raw SMILES -> conformers -> JEPA checkpoint -> CSV", flush=True)
        smiles_csv.write_text("smiles\nnot_a_molecule\n")
        bad_output = work / "invalid.csv"
        run("predict.py", "--checkpoint", checkpoints["jepa"],
            "--input-csv", smiles_csv, "--output", bad_output, success=False)
        assert not bad_output.exists()
        broken = work / "broken.pt"
        torch.save({"model": model.state_dict(), "target_mean": 0, "target_std": 0}, broken)
        try:
            load_model(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid scaling accepted")
        corrupted = dict(macrocycle)
        corrupted["z_2d"] = macrocycle["z_2d"].clone()
        corrupted["z_2d"][0] = 7
        try:
            collate([(corrupted, {"y": 0})])
        except RuntimeError:
            pass
        else:
            raise AssertionError("Atom mismatch accepted")
        state = model.state_dict()
        del state["head.1.weight"]
        torch.save({"model": state, "target_mean": mean, "target_std": std}, broken)
        try:
            load_model(broken)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Incomplete model checkpoint accepted")
        assert np.isnan(metrics([1], [2])["pearson"])
        print("PASS: invalid SMILES, scaling, atom mismatch, incomplete checkpoint and singleton metrics", flush=True)
    print("ALL CHECKS PASSED (toy data; no paper checkpoint accuracy claim)", flush=True)


if __name__ == "__main__":
    main()
