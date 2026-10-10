"""Kiểm các lỗi có thể làm sai bài toán đa nhãn; không tải pretrained model."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from src.models.distilbert_study import ROOT, format_predictions, predict_texts, prepare_frames, read_json, validate_config, write_json
from scripts.distilbert.summarize_distilbert import summarize


class C3Tests(unittest.TestCase):
    def test_multilabel_loss_is_bce_and_roundtrip(self):
        torch.manual_seed(42)
        config = DistilBertConfig(vocab_size=32, dim=16, hidden_dim=24, n_layers=1,
                                 n_heads=2, num_labels=28, problem_type="multi_label_classification",
                                 dropout=0, attention_dropout=0, seq_classif_dropout=0)
        model = DistilBertForSequenceClassification(config).eval()
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        labels = torch.zeros((2, 28), dtype=torch.float32)
        labels[0, [1, 7]] = 1  # Hai nhãn đồng thời, không phải class index.
        output = model(input_ids=tokens, labels=labels)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(output.logits, labels)
        torch.testing.assert_close(output.loss, expected)
        output.loss.backward()
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = DistilBertForSequenceClassification.from_pretrained(directory).eval()
            torch.testing.assert_close(restored(input_ids=tokens).logits, output.logits)

    def test_threshold_can_return_multiple_or_no_labels(self):
        names = ["joy", "love", "neutral"]
        result = format_predictions(["a", "b"], [[.5, .8, .1], [.1, .2, .3]], names, [4, 150], 128, .5)
        self.assertEqual(result[0]["labels"], ["joy", "love"])
        self.assertEqual(result[1]["labels"], [])
        self.assertTrue(result[1]["truncated"])

    def test_invalid_scores_rejected(self):
        for scores in ([[float("nan")]], [[1.1]], [[-.1]], [[.1, .2]]):
            with self.assertRaises(ValueError):
                format_predictions(["a"], scores, ["joy"], [4], 128, .5)

    def test_empty_or_excessively_long_input_rejected_before_model(self):
        for texts in ([], ["  "], [None], ["a" * 20001]):
            with self.assertRaises(ValueError):
                predict_texts({}, texts)

    def test_full_cannot_use_pilot_subset(self):
        config = read_json(ROOT / "configs/distilbert_pilot.json")
        config["mode"] = "full"
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_data_loader_reads_train_validation_only(self):
        config = read_json(ROOT / "configs/distilbert_pilot.json")
        with patch("src.models.distilbert_study.load_goemotions", side_effect=RuntimeError("stop before IO")) as loader:
            with self.assertRaises(RuntimeError):
                prepare_frames(config)
            self.assertEqual(loader.call_args.kwargs["splits"], ("train", "validation"))
            self.assertFalse(loader.call_args.kwargs["write_metadata"])

    def test_aggregate_rejects_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory)/"run.json", {"status": "complete", "mode": "pilot"})
            with self.assertRaisesRegex(ValueError, "pilot"):
                summarize([directory])

    def test_sample_std_and_distinct_seed_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for seed, value in zip((42, 123, 2026), (.1, .2, .3)):
                path = Path(directory)/str(seed)
                paths.append(path)
                write_json(path/"run.json", {"status":"complete", "mode":"full", "n_train":43410,
                    "n_validation":5426, "test_used":False, "artifact_sha256":{}, "config":{"seed":seed},
                    "data_sha256":{}, "label_names":["joy"], "source_sha256":{}, "best_epoch":1})
                write_json(path/"validation_metrics.json", {k:value for k in ("macro_f1", "micro_f1",
                    "macro_precision", "macro_recall", "micro_precision", "micro_recall", "hamming_loss")})
            _, stats = summarize(paths)
            np.testing.assert_allclose(stats["mean"], .2)
            np.testing.assert_allclose(stats["sample_std"], .1)
            with self.assertRaisesRegex(ValueError, "Seed"):
                summarize(paths+[paths[0]])
            with self.assertRaisesRegex(ValueError, "ba seed"):
                summarize(paths[:2])


if __name__ == "__main__":
    unittest.main()
