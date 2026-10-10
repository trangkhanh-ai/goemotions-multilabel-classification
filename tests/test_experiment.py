"""Kiểm ranh giới validation/test và nhận diện checkpoint, không tải model."""
import unittest
from unittest.mock import patch
import numpy as np

from src.datasets.goemotions import REVISION
from src.evaluation.protocols import validate_protocol


class ProtocolTests(unittest.TestCase):
    def protocol(self):
        return {"protocol_version": 1, "smoke": False, "method": "C",
                "data_revision": REVISION, "label_names": ["a", "b"],
                "checkpoint": "model", "model_revision": "abc",
                "run_metadata_sha256": "digest", "validation_scores_sha256": "digest",
                "threshold_mode": "tuned", "thresholds": [0.2, 0.3],
                "configurations": [{"threshold_mode": "fixed", "thresholds": 0.5},
                                   {"threshold_mode": "global", "thresholds": 0.4},
                                   {"threshold_mode": "tuned", "thresholds": [0.2, 0.3]}]}

    def validate(self, value):
        metadata = {"method": "C", "checkpoint": "model", "model_revision": "abc"}
        with patch("src.evaluation.protocols.check_full_run", return_value=metadata), \
             patch("src.evaluation.protocols.sha256", return_value="digest"):
            return validate_protocol("fake", value, ["a", "b"])

    def test_complete_protocol(self):
        self.assertEqual(self.validate(self.protocol())["method"], "C")

    def test_threshold_change_rejected(self):
        value = self.protocol()
        value["thresholds"] = [0.2, 0.8]
        with self.assertRaises(ValueError):
            self.validate(value)

    def test_checkpoint_change_rejected(self):
        value = self.protocol()
        value["checkpoint"] = "different"
        with self.assertRaises(ValueError):
            self.validate(value)

    def test_missing_comparison_rejected(self):
        value = self.protocol()
        value["configurations"].pop(0)
        with self.assertRaises(ValueError):
            self.validate(value)


if __name__ == "__main__":
    unittest.main()
