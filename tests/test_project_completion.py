"""Kiểm ghi scores bị ngắt và bảng seed: dữ liệu nhỏ, không tải mô hình/GPU."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from scripts.transformers import evaluate_transformer_test
from scripts.analysis import summarize_project
from src.evaluation.protocols import save_json
from src.models.transformer import METRIC_NAMES, run_folder


class ProjectCompletionTest(unittest.TestCase):
    def test_interrupted_test_score_write_can_retry_same_protocol(self):
        """Ngắt ngay lúc ghi NPZ: không để ZIP dở mang tên file hoàn tất."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "run"
            folder.mkdir()
            labels = ["joy", "neutral"]
            save_json(root / "data/labels.json", labels)
            protocol_path = folder / "final_protocol.json"
            save_json(protocol_path, {"configurations": [
                {"threshold_mode": "fixed", "thresholds": 0.5},
                {"threshold_mode": "global", "thresholds": 0.4},
                {"threshold_mode": "tuned", "thresholds": [0.4, 0.6]},
            ]})
            frame = pd.DataFrame({"id": ["first", "second"], "text": ["happy", "meeting"],
                                  "labels": [[0], [1]]})
            metadata = {"architecture": "bert", "seed": 42, "config": {"max_length": 128}}
            scores = np.array([[0.9, 0.1], [0.1, 0.9]])
            model = SimpleNamespace(to=lambda target: "mock-model")
            torch = SimpleNamespace(set_num_threads=Mock(), utils=SimpleNamespace(
                data=SimpleNamespace(DataLoader=Mock(return_value="mock-loader"))))
            transformers = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value="mock-tokenizer")),
                AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=Mock(return_value=model)),
            )

            def interrupt_write(stream, **arrays):
                stream.write(b"interrupted ZIP bytes")
                raise OSError("write interrupted")

            with (patch.object(evaluate_transformer_test, "ROOT", root),
                  patch.object(evaluate_transformer_test, "validate_protocol", return_value=metadata),
                  patch.object(evaluate_transformer_test, "load_goemotions", return_value=({"test": frame}, labels, {})),
                  patch.dict("sys.modules", {"torch": torch, "transformers": transformers}),
                  patch("src.models.transformer.resolve_device", return_value="cpu"),
                  patch("scripts.transformers.train_transformer.encoded_dataset", return_value=("mock-dataset", None)),
                  patch("scripts.transformers.train_transformer.predict_scores", return_value=scores) as predictor):
                with patch("src.models.zero_shot.np.savez_compressed", side_effect=interrupt_write):
                    with self.assertRaisesRegex(OSError, "write interrupted"):
                        evaluate_transformer_test.evaluate_run(folder, protocol_path, device="cpu")
                self.assertFalse((folder / "test_scores.npz").exists())
                self.assertFalse((folder / "test_results.json").exists())
                self.assertTrue((folder / "test_scores.npz.part").exists())
                result = evaluate_transformer_test.evaluate_run(folder, protocol_path, device="cpu")
                self.assertEqual(result["results"][0]["metrics"]["macro_f1"], 1.0)
                self.assertFalse((folder / "test_scores.npz.part").exists())
                with np.load(folder / "test_scores.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["scores"], scores)
                    self.assertEqual(saved["ids"].tolist(), ["first", "second"])
                cached = evaluate_transformer_test.evaluate_run(folder, protocol_path, device="cpu")
                self.assertEqual(cached, result)
                self.assertEqual(predictor.call_count, 2)

    def make_seed_runs(self, root):
        """Run fixtures đã hoàn tất; collector không cần model hay dữ liệu thật."""
        save_json(root / "data/labels.json", ["joy", "neutral"])
        runs = {}
        for seed, f1 in zip(summarize_project.SEEDS, [0.3, 0.5, 0.7]):
            folder = run_folder(root, "bert", seed)
            save_json(folder / "run_metadata.json", {"completed": True})
            runs[folder] = {"config": {"learning_rate": 5e-5, "epochs": 4},
                            "model_revision": "a" * 40,
                            "metrics": {"n_samples": 5426,
                                        **{name: f1 for name in METRIC_NAMES}}}
        return runs

    def test_standalone_collector_rejects_mixed_config_or_checkpoint_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = self.make_seed_runs(root)
            changed_folder = run_folder(root, "bert", 123)
            for change in ("config", "model_revision"):
                with self.subTest(change=change):
                    runs = copy.deepcopy(reference)
                    if change == "config":
                        runs[changed_folder]["config"]["learning_rate"] = 1e-3
                    else:
                        runs[changed_folder]["model_revision"] = "b" * 40
                    with patch.object(summarize_project, "load_transformer_run",
                                      side_effect=lambda folder, **kwargs: runs[folder]):
                        with self.assertRaisesRegex(ValueError, "các seed khác cấu hình hoặc revision"):
                            summarize_project.collect(root)

    def test_matching_seed_runs_keep_original_mean_and_sample_std(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = self.make_seed_runs(root)
            with patch.object(summarize_project, "load_transformer_run",
                              side_effect=lambda folder, **kwargs: runs[folder]):
                records, _, _ = summarize_project.collect(root)
            rows = summarize_project.summarize(records)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["n_runs"], 3)
            self.assertAlmostEqual(rows[0]["macro_f1_mean"], 0.5)
            self.assertAlmostEqual(rows[0]["macro_f1_std"], 0.2)


if __name__ == "__main__":
    unittest.main()
