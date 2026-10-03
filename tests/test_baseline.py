"""Kiểm các lỗi có thể làm sai kết quả khi ghép ID và chọn ngưỡng."""

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from src.baseline import (load_aligned_scores, tune_thresholds, tune_global_threshold,
                          label_error_pairs, load_run_metadata, load_thresholds)
from src.data import REVISION, sha256
from scripts.run_baseline import build_model


class BaselineTest(unittest.TestCase):
    def test_scores_are_joined_by_id(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scores.npz"
            np.savez(path, ids=np.array(["b", "a"]),
                     scores=np.array([[0.2, 0.8], [0.9, 0.1]]),
                     label_names=np.array(["x", "y"]))
            aligned = load_aligned_scores(path, ["a", "b"], ["x", "y"])
            np.testing.assert_allclose(aligned, [[0.9, 0.1], [0.2, 0.8]])
            with self.assertRaises(ValueError):
                load_aligned_scores(path, ["a", "c"], ["x", "y"])

    def test_tuning_uses_each_label_separately(self):
        truth = np.array([[1, 0], [0, 1], [0, 0]])
        scores = np.array([[0.4, 0.2], [0.2, 0.8], [0.1, 0.1]])
        thresholds = tune_thresholds(truth, scores)
        self.assertEqual(thresholds.shape, (2,))
        self.assertLessEqual(thresholds[0], 0.4)
        self.assertGreater(thresholds[0], 0.2)
        self.assertLessEqual(thresholds[1], 0.8)
        self.assertGreater(thresholds[1], 0.2)

    def test_global_threshold_maximizes_macro_on_declared_grid(self):
        truth = np.array([[1, 0], [0, 1], [0, 0]])
        scores = np.array([[0.4, 0.2], [0.2, 0.8], [0.1, 0.1]])
        chosen, curve = tune_global_threshold(truth, scores)
        best = next(row for row in curve if row["threshold"] == chosen)
        self.assertEqual(best["macro_f1"], 1)
        self.assertGreater(chosen, 0.2)
        self.assertLessEqual(chosen, 0.4)

    def test_tuning_does_not_change_default_when_all_candidates_tie(self):
        truth = np.zeros((3, 2), dtype=int)
        scores = np.zeros((3, 2))
        np.testing.assert_array_equal(tune_thresholds(truth, scores), [0.5, 0.5])
        self.assertEqual(tune_global_threshold(truth, scores)[0], 0.5)

    def test_tuning_rejects_non_probability_input(self):
        with self.assertRaises(ValueError):
            tune_thresholds(np.array([[1, 0]]), np.array([[np.nan, 0.2]]))

    def test_pairs_count_missed_and_extra_labels_in_same_sample(self):
        truth = np.array([[1, 0], [1, 1], [0, 1]])
        scores = np.array([[0.1, 0.9], [0.1, 0.1], [0.9, 0.1]])
        pairs = label_error_pairs(truth, scores, ["a", "b"])
        self.assertEqual({(r["missed_label"], r["extra_label"]): r["count"] for r in pairs},
                         {("a", "b"): 1, ("b", "a"): 1})

    def test_validation_transform_cannot_extend_training_vocabulary(self):
        model = build_model()
        model.fit(["shared love", "shared love", "shared hate", "shared hate"],
                  np.array([[1, 0], [1, 0], [0, 1], [0, 1]]))
        before = dict(model.named_steps["tfidf"].vocabulary_)
        model.predict_proba(["validationonly love"])
        self.assertEqual(before, model.named_steps["tfidf"].vocabulary_)
        self.assertNotIn("validationonly", before)

    def test_old_thresholds_and_mixed_model_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "model.joblib").write_bytes(b"model one")
            (folder / "validation_scores.npz").write_bytes(b"scores one")
            metadata = {"artifact_version": 2, "variant": "standard", "smoke": False,
                        "data_revision": REVISION, "label_names": ["a", "b"],
                        "artifact_sha256": {name: sha256(folder / name)
                                            for name in ("model.joblib", "validation_scores.npz")}}
            (folder / "validation_metrics.json").write_text(json.dumps(metadata))
            saved = {key: metadata[key] for key in
                     ("variant", "data_revision", "label_names", "artifact_sha256")}
            saved.update({"thresholds": [0.4, 0.6], "global_threshold": 0.4})
            (folder / "thresholds_validation.json").write_text(json.dumps(saved))
            load_run_metadata(folder, "standard", ["a", "b"])
            np.testing.assert_array_equal(load_thresholds(folder, metadata, "tuned"), [0.4, 0.6])
            (folder / "model.joblib").write_bytes(b"model two")
            with self.assertRaises(ValueError):
                load_run_metadata(folder, "standard", ["a", "b"])
            metadata["artifact_sha256"]["model.joblib"] = sha256(folder / "model.joblib")
            with self.assertRaises(ValueError):
                load_thresholds(folder, metadata, "tuned")


if __name__ == "__main__":
    unittest.main()
