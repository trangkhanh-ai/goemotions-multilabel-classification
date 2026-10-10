"""Kiểm luồng test bằng dữ liệu GIẢ trong thư mục tạm; không mở GoEmotions test."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.baseline.evaluate_baseline_test import evaluate_frozen_protocol
from src.datasets.goemotions import REVISION, sha256


class FinalProtocolTest(unittest.TestCase):
    def test_missing_protocol_fails_before_test_data_are_opened(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("scripts.baseline.evaluate_baseline_test.load_goemotions") as loader:
                with self.assertRaises(FileNotFoundError):
                    evaluate_frozen_protocol(Path(temporary))
                loader.assert_not_called()

    def test_changed_model_fails_before_test_data_are_opened(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "data/processed/baseline"
            folder.mkdir(parents=True)
            (root / "data/labels.json").write_text(json.dumps(["a", "b"]))
            protocol = {"schema_version": 1, "data_revision": REVISION,
                        "label_names": ["a", "b"],
                        "runs": {v: {"folder": v, "artifact_sha256": {"model.joblib": "old"}}
                                 for v in ("standard", "balanced")}}
            (folder / "final_protocol.json").write_text(json.dumps(protocol))
            with (patch("scripts.baseline.evaluate_baseline_test.load_goemotions") as loader,
                  patch("scripts.baseline.evaluate_baseline_test.load_run_metadata",
                        return_value={"artifact_sha256": {"model.joblib": "changed"}})):
                with self.assertRaises(ValueError):
                    evaluate_frozen_protocol(root)
                loader.assert_not_called()

    def test_frozen_batch_retains_original_baseline_and_reuses_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            labels = ["a", "b"]
            (root / "data/labels.json").write_text(json.dumps(labels))
            runs = {}
            for variant in ("standard", "balanced"):
                folder = root / variant
                folder.mkdir()
                (folder / "thresholds_validation.json").write_text("{}")
                runs[variant] = {"folder": variant, "artifact_sha256": {},
                                 "threshold_file_sha256": sha256(folder / "thresholds_validation.json")}
            configs = [{"name": f"{v}_{m}", "variant": v, "threshold_mode": m,
                        "thresholds": [0.5, 0.5]}
                       for v in runs for m in ("fixed", "global", "tuned")]
            protocol = {"schema_version": 1, "data_revision": REVISION, "label_names": labels,
                        "runs": runs, "configurations": configs,
                        "rare_label_ids_from_train": [0, 1],
                        "selected_configuration": "balanced_tuned"}
            location = root / "data/processed/baseline"
            location.mkdir(parents=True)
            (location / "final_protocol.json").write_text(json.dumps(protocol))
            frame = pd.DataFrame({"text": ["first", "second"], "labels": [[0], [1]],
                                  "id": ["x", "y"]})
            model = SimpleNamespace(predict_proba=lambda _: np.array([[0.9, 0.1], [0.1, 0.9]]))
            with (patch("scripts.baseline.evaluate_baseline_test.load_goemotions", return_value=({"test": frame}, labels, {})) as loader,
                  patch("scripts.baseline.evaluate_baseline_test.load_run_metadata", return_value={"artifact_sha256": {}}),
                  patch("scripts.baseline.evaluate_baseline_test.joblib.load", return_value=model)):
                first = evaluate_frozen_protocol(root)
                second = evaluate_frozen_protocol(root)
                self.assertEqual(len(first["results"]), 6)
                self.assertIn("standard_fixed", [r["name"] for r in first["results"]])
                self.assertEqual(first["selected_on_validation"], "balanced_tuned")
                self.assertEqual(first, second)
                self.assertEqual(loader.call_count, 1)


if __name__ == "__main__":
    unittest.main()
