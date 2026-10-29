"""Run with python -m unittest discover -s tests -p test_protocol.py."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from ncmol.protocol import audit_splits, initialization_exposure, molecule_id
from summarize_results import add_result


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.payload = {"graphs": {}, "splits": {"random": {}}}
        for split, smiles in zip(("train", "val", "test"), ("CCO", "CCN", "CCC")):
            self.payload["graphs"][split] = {"smiles": smiles}
            self.payload["splits"]["random"][split] = [{"key": split, "smiles": smiles, "y": 0}]

    def test_equivalent_smiles_and_maps(self):
        self.assertEqual(molecule_id("CCO"), molecule_id("OCC"))
        self.assertEqual(molecule_id("CCO"), molecule_id("[CH3:1]CO"))

    def test_alias_keys_cannot_hide_overlap(self):
        self.payload["graphs"]["test"]["smiles"] = "OCC"
        self.payload["splits"]["random"]["test"][0]["smiles"] = "OCC"
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit_splits(self.payload, "random")

    def test_graph_mismatch(self):
        self.payload["graphs"]["test"]["smiles"] = "CO"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            audit_splits(self.payload, "random")

    def test_duplicates(self):
        rows = self.payload["splits"]["random"]["train"]
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_splits(self.payload, "random")

    def test_source_selection_and_training_exposure(self):
        ids = audit_splits(self.payload, "random")
        for split in ("val", "test"):
            source = {"protocol": {"version": 1, "selection_exposure_ids": ids[split]}}
            with self.assertRaisesRegex(ValueError, "overlap"):
                initialization_exposure(source, ids)
        source = {"protocol": {"version": 1, "selection_exposure_ids": ids["train"]}}
        self.assertEqual(initialization_exposure(source, ids), set(ids["train"]))

    def test_legacy_provenance_rejected(self):
        with self.assertRaisesRegex(ValueError, "provenance"):
            initialization_exposure({}, audit_splits(self.payload, "random"))

    def test_cross_task_overlap_and_same_task_ancestry(self):
        ids = audit_splits(self.payload, "random")
        source = {"protocol": {"version": 2, "selection_exposure_ids": ids["test"],
                               "exposure_by_task": {"pampa": ids["test"]}}}
        self.assertEqual(initialization_exposure(source, ids, "cross-task", "caco_ab"), set(ids["test"]))
        with self.assertRaisesRegex(ValueError, "overlap"):
            initialization_exposure(source, ids, "cross-task", "pampa")
        with self.assertRaisesRegex(ValueError, "overlap"):
            initialization_exposure(source, ids)
        source["protocol"]["exposure_by_task"] = {"unspecified": ids["test"]}
        with self.assertRaisesRegex(ValueError, "task IDs"):
            initialization_exposure(source, ids, "cross-task", "caco_ab")

    def test_summary_preserves_all_models_regardless_of_test_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scores in ((0.1, 9.0), (9.0, 0.1)):
                rows = []
                for name, score in zip(("base", "jepa"), scores):
                    path = root / name
                    path.mkdir(exist_ok=True)
                    test = {key: score for key in ("rmse", "mae", "r2", "pearson", "spearman", "ci")}
                    (path / "metrics.json").write_text(json.dumps(
                        {"best_epoch": 1, "val": {"rmse": 0.5}, "test": test}))
                    add_result(rows, "direct_cell", "toy", "random", name, path)
                self.assertEqual([r["variant"] for r in rows], ["base", "jepa"])


if __name__ == "__main__":
    unittest.main()
