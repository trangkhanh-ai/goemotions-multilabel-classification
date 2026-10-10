"""Các kiểm tra B dùng điểm GIẢ; không tải BART, không mở GoEmotions test."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts.zero_shot.run_zero_shot import run
from src.datasets.goemotions import EXPECTED_ROWS, EXPECTED_SHA256, REVISION, sha256
from src.models.zero_shot import (CHECKPOINT, HYPOTHESIS_TEMPLATE, map_pipeline_scores,
                           predict_in_batches, validate_test_protocol, write_json, write_scores)


class ZeroShotTest(unittest.TestCase):
    def test_sorted_pipeline_labels_are_mapped_to_canonical_columns(self):
        result = [{"labels": ["anger", "joy"], "scores": [0.9, 0.2]},
                  {"labels": ["joy", "anger"], "scores": [0.8, 0.1]}]
        np.testing.assert_allclose(map_pipeline_scores(result, ["joy", "anger"]),
                                   [[0.2, 0.9], [0.8, 0.1]])

    def test_missing_duplicate_unknown_and_nonfinite_labels_are_rejected(self):
        bad = [{"labels": ["joy"], "scores": [0.2]},
               {"labels": ["joy", "joy"], "scores": [0.2, 0.1]},
               {"labels": ["joy", "other"], "scores": [0.2, 0.1]},
               {"labels": ["joy", "anger"], "scores": [float("nan"), 0.1]},
               {"labels": ["joy", "anger"], "scores": [1.1, 0.1]}]
        for result in bad:
            with self.subTest(result=result), self.assertRaises(ValueError):
                map_pipeline_scores(result, ["joy", "anger"])

    def test_resume_keeps_completed_batches_and_recomputes_only_unfinished(self):
        with tempfile.TemporaryDirectory() as folder:
            def one_batch(texts):
                return [{"labels": ["b", "a"], "scores": [0.1, 0.8]} for _ in texts]
            failing = Mock(side_effect=[one_batch(["one", "two"]), RuntimeError("interrupted")])
            with self.assertRaises(RuntimeError):
                predict_in_batches(failing, ["one", "two", "three"], ["1", "2", "3"],
                                   ["a", "b"], folder, {"revision": "abc"}, batch_size=2)
            manifest = json.loads((Path(folder) / "checkpoint_manifest.json").read_text())
            self.assertEqual(manifest["chunks"][0]["stop"], 2)
            resumed = Mock(side_effect=one_batch)
            scores, finished = predict_in_batches(resumed, ["one", "two", "three"], ["1", "2", "3"],
                                                  ["a", "b"], folder, {"revision": "abc"}, batch_size=2)
            resumed.assert_called_once_with(["three"])
            np.testing.assert_allclose(scores, [[0.8, 0.1]] * 3)
            self.assertEqual(finished["status"], "complete")
            completed = Mock(side_effect=RuntimeError("should not run"))
            again, _ = predict_in_batches(completed, ["one", "two", "three"], ["1", "2", "3"],
                                         ["a", "b"], folder, {"revision": "abc"}, batch_size=2)
            completed.assert_not_called()
            np.testing.assert_array_equal(scores, again)

    def test_resume_rejects_changed_revision_text_order_and_batch_size(self):
        with tempfile.TemporaryDirectory() as folder:
            predictor = Mock(return_value=[{"labels": ["a"], "scores": [0.9]}])
            predict_in_batches(predictor, ["one"], ["1"], ["a"], folder, {"revision": "abc"}, 1)
            for texts, config, batch in [(["one"], {"revision": "xyz"}, 1),
                                          (["changed"], {"revision": "abc"}, 1),
                                          (["one"], {"revision": "abc"}, 2)]:
                with self.subTest(texts=texts, config=config, batch=batch), self.assertRaises(ValueError):
                    predict_in_batches(predictor, texts, ["1"], ["a"], folder, config, batch)

    def test_resume_rejects_corrupted_saved_chunk(self):
        with tempfile.TemporaryDirectory() as folder:
            predict_in_batches(lambda _: [{"labels": ["a"], "scores": [0.9]}],
                               ["one"], ["1"], ["a"], folder, {}, 1)
            next(Path(folder).glob("chunk_*.npz")).write_bytes(b"broken")
            with self.assertRaises(ValueError):
                predict_in_batches(Mock(), ["one"], ["1"], ["a"], folder, {}, 1)

    def test_incomplete_pipeline_batch_is_not_committed(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                predict_in_batches(lambda _: [{"labels": ["a"], "scores": [0.9]}],
                                   ["one", "two"], ["1", "2"], ["a"], folder, {}, 2)
            manifest = json.loads((Path(folder) / "checkpoint_manifest.json").read_text())
            self.assertEqual(manifest["chunks"], [])
            self.assertEqual(list(Path(folder).glob("chunk_*.npz")), [])

    def test_protocol_hash_and_thresholds_are_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels = ["a", "b"]
            count = EXPECTED_ROWS["validation"]
            write_scores(root / "validation_scores.npz", [str(i) for i in range(count)],
                         np.tile([[0.8, 0.1]], (count, 1)), labels)
            metadata = {"method": "zero_shot", "split": "validation", "status": "complete", "smoke": False,
                        "checkpoint": CHECKPOINT, "model_revision": "a" * 40,
                        "hypothesis_template": HYPOTHESIS_TEMPLATE, "label_names": labels,
                        "data_revision": REVISION, "sample_count": EXPECTED_ROWS["validation"],
                        "data_sha256": {"validation": EXPECTED_SHA256["validation"]}, "multi_label": True,
                        "artifact_sha256": {"validation_scores.npz": sha256(root / "validation_scores.npz")}}
            write_json(root / "run_metadata.json", metadata)
            protocol = dict(metadata, protocol_version=1, threshold_mode="tuned", thresholds=[0.4, 0.6],
                            frozen_at_utc="2026-10-08T00:00:00+00:00",
                            validation_scores_sha256=sha256(root / "validation_scores.npz"),
                            run_metadata_sha256=sha256(root / "run_metadata.json"))
            write_json(root / "protocol.json", protocol)
            accepted, _ = validate_test_protocol(root / "protocol.json", root, labels)
            self.assertEqual(accepted["thresholds"], [0.4, 0.6])
            protocol["thresholds"] = [float("nan"), 0.6]
            write_json(root / "protocol.json", protocol)
            with self.assertRaises(ValueError):
                validate_test_protocol(root / "protocol.json", root, labels)
            protocol["thresholds"] = [0.4, 0.6]
            write_json(root / "protocol.json", protocol)
            (root / "validation_scores.npz").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_test_protocol(root / "protocol.json", root, labels)

    def test_test_without_protocol_fails_before_loading_data_or_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            write_json(root / "data/labels.json", [f"label_{i}" for i in range(28)])
            args = argparse.Namespace(split="test", device="cpu", batch_size=1, smoke=False,
                                      limit=None, protocol=None, revision="main")
            with (patch("scripts.zero_shot.run_zero_shot.ROOT", root),
                  patch("scripts.zero_shot.run_zero_shot.load_goemotions") as loader,
                  patch("scripts.zero_shot.run_zero_shot.build_pipeline") as model):
                with self.assertRaises(ValueError):
                    run(args)
                loader.assert_not_called()
                model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
