"""Phân tích lỗi dùng dữ liệu GIẢ; không huấn luyện và không mở GoEmotions test."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.analysis.analyze_project_errors import (analyze_arrays, analyze_project, checked_threshold,
                                            error_masks, representative_run)
from src.datasets.goemotions import REVISION, sha256
from src.evaluation.protocols import save_json


class ProjectErrorsTest(unittest.TestCase):
    def test_three_error_groups_have_distinct_definitions(self):
        truth = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        scores = np.array([[0.1, 0.9, 0.1, 0.1], [0.1, 0.1, 0.9, 0.1],
                           [0.1, 0.1, 0.9, 0.1], [0.1, 0.1, 0.1, 0.9]])
        masks, _, missed, extra = error_masks(truth, scores, [0])
        np.testing.assert_array_equal(masks["partial_multi_label"], [True, False, False, False])
        np.testing.assert_array_equal(masks["rare_false_negative"], [True, True, False, False])
        np.testing.assert_array_equal(masks["missed_extra_pair"], [False, True, True, False])
        self.assertEqual(int(missed.sum()), 3)
        self.assertEqual(int(extra.sum()), 2)

    def test_representative_seed_is_selected_on_validation_and_tie_uses_small_seed(self):
        runs = [{"seed": 42, "metrics": {"macro_f1": 0.5}, "test_f1": 0.8},
                {"seed": 123, "metrics": {"macro_f1": 0.6}, "test_f1": 0.2},
                {"seed": 2026, "metrics": {"macro_f1": 0.6}, "test_f1": 0.9}]
        self.assertEqual(representative_run(runs)["seed"], 123)
        with self.assertRaises(ValueError):
            representative_run([runs[0], runs[0]])

    def test_same_examples_for_all_architectures_and_linguistic_notes_stay_empty(self):
        labels = ["rare", "joy", "anger", "neutral"]
        frame = pd.DataFrame({"id": ["d", "c", "b", "a"], "text": ["one", "two", "three", "four"],
                              "labels": [[0, 1], [0], [1], [3]]})
        runs = {name: {"seed": 42} for name in ("bert", "roberta", "distilbert")}
        good = np.array([[0.9, 0.9, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1],
                         [0.1, 0.9, 0.1, 0.1], [0.1, 0.1, 0.1, 0.9]])
        bad = np.array([[0.1, 0.9, 0.1, 0.1], [0.1, 0.1, 0.9, 0.1],
                        [0.1, 0.1, 0.9, 0.1], [0.1, 0.1, 0.1, 0.9]])
        arrays = {"bert": bad, "roberta": good, "distilbert": good}
        counts, pairs, rare, examples = analyze_arrays(frame, labels, [0], [2, 2, 0, 1],
                                                      runs, arrays, {name: 0.5 for name in runs}, 1)
        bert_rare = next(row for row in counts if row["architecture"] == "bert" and row["category"] == "rare_false_negative")
        self.assertEqual(bert_rare["error_samples"], 2)
        self.assertEqual(bert_rare["eligible_samples"], 2)
        self.assertEqual(bert_rare["fraction_eligible"], 1.0)
        pair = next(row for row in pairs if row["missed_label"] == "rare")
        self.assertEqual(pair["extra_label"], "anger")
        self.assertEqual(pair["sample_count"], 1)
        self.assertEqual(pair["fraction_of_missed_label_fn"], 0.5)
        self.assertEqual(next(row for row in rare if row["architecture"] == "bert")["fn"], 2)
        selected = [row for row in examples if row["category"] == "rare_false_negative"]
        self.assertEqual({row["id"] for row in selected}, {"c"})
        self.assertEqual({row["architecture"] for row in selected}, set(runs))
        self.assertTrue(all(row["manual_linguistic_notes"] == "" for row in examples))
        self.assertFalse(next(row for row in selected if row["architecture"] == "roberta")["error_present"])

    def test_one_sentence_can_contribute_multiple_fn_fp_pairs(self):
        labels = ["a", "b", "c", "d"]
        frame = pd.DataFrame({"id": ["x"], "text": ["example"], "labels": [[0, 1]]})
        _, pairs, _, _ = analyze_arrays(frame, labels, [0], [1, 1, 0, 0],
                                       {"bert": {"seed": 42}}, {"bert": np.array([[0.1, 0.1, 0.9, 0.9]])},
                                       {"bert": 0.5})
        self.assertEqual(len(pairs), 4)
        self.assertTrue(all(row["sample_count"] == 1 for row in pairs))

    def test_test_without_protocol_fails_before_opening_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            save_json(root / "data/labels.json", ["a", "b"])
            run = {"run_dir": "representative", "seed": 42, "architecture": "bert", "label_names": ["a", "b"]}
            with (patch("scripts.analysis.analyze_project_errors.load_representatives", return_value=({"bert": run}, [])),
                  patch("scripts.analysis.analyze_project_errors.load_goemotions") as loader):
                with self.assertRaises(FileNotFoundError):
                    analyze_project(root, split="test")
                loader.assert_not_called()

    def test_changed_test_hash_is_rejected_before_reading_test_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "run"
            folder.mkdir()
            run = {"run_dir": "run", "seed": 42, "architecture": "bert",
                   "checkpoint": "model", "model_revision": "a" * 40}
            protocol = {"checkpoint": "model", "model_revision": "a" * 40, "seed": 42,
                        "architecture": "bert", "selection_data": "validation",
                        "frozen_at_utc": "2026-10-08T00:00:00+00:00"}
            save_json(folder / "final_protocol.json", protocol)
            (folder / "test_scores.npz").write_bytes(b"first scores")
            save_json(folder / "test_results.json", {"split": "test", "method": "C", "architecture": "bert",
                                                     "seed": 42, "protocol_sha256": sha256(folder / "final_protocol.json"),
                                                     "scores_sha256": "wrong", "results": [{"threshold_mode": "fixed"}]})
            with patch("scripts.analysis.analyze_project_errors.validate_protocol"):
                with self.assertRaises(ValueError):
                    checked_threshold(root, run, ["a", "b"], "test", "fixed")

    def test_nonfinite_scores_and_invalid_rare_mapping_are_rejected(self):
        truth = np.array([[1, 0]])
        with self.assertRaises(ValueError):
            error_masks(truth, [[float("nan"), 0.1]], [0])
        with self.assertRaises(ValueError):
            error_masks(truth, [[0.8, 0.1]], [2])

    def test_synthetic_end_to_end_writes_six_reviewable_files_without_test_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            labels = ["a", "b"]
            save_json(root / "data/labels.json", labels)
            runs = {}
            for architecture in ("bert", "roberta", "distilbert"):
                folder = root / architecture
                folder.mkdir()
                save_json(folder / "run_metadata.json", {"seed": 42})
                (folder / "validation_scores.npz").write_bytes(b"synthetic placeholder")
                runs[architecture] = {"seed": 42, "architecture": architecture,
                                      "label_names": labels, "run_dir": architecture,
                                      "metrics": {"macro_f1": 0.5}}
            train = pd.DataFrame({"id": ["t1", "t2"], "text": ["train one", "train two"], "labels": [[0], [1]]})
            validation = pd.DataFrame({"id": ["v1", "v2"], "text": ["val one", "val two"], "labels": [[0, 1], [0]]})
            loader_result = ({"train": train, "validation": validation}, labels,
                             {"files": [{"split": "train", "sha256": "train hash"},
                                        {"split": "validation", "sha256": "validation hash"}]})
            with (patch("scripts.analysis.analyze_project_errors.load_representatives", return_value=(runs, [])),
                  patch("scripts.analysis.analyze_project_errors.load_goemotions", return_value=loader_result) as loader,
                  patch("scripts.analysis.analyze_project_errors.load_aligned_scores", return_value=np.array([[0.8, 0.1], [0.1, 0.8]]))):
                manifest = analyze_project(root)
                loader.assert_called_once_with(root, write_metadata=False, splits=("train", "validation"))
            self.assertEqual(manifest["split"], "validation")
            destination = root / "reports/errors_validation_standard_fixed"
            self.assertEqual({path.name for path in destination.iterdir()},
                             {"counts.csv", "pairs.csv", "rare_fn.csv", "examples.csv", "summary.md", "manifest.json"})
            self.assertEqual(json.loads((destination / "manifest.json").read_text(encoding="utf-8"))["sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
