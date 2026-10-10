"""Kiểm logic C/D bằng dữ liệu giả trong test, không tải checkpoint mạng."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.datasets.goemotions import EXPECTED_ROWS, REVISION, sha256
from src.models.transformer import (METRIC_NAMES, artifact_hashes, load_demo_selection,
                        load_transformer_run, multilabel_loss, positive_weights,
                        summarize_architectures, training_config, write_json)


def example_run(architecture, seed, macro_f1):
    labels = [f"label_{i}" for i in range(28)]
    return {"artifact_version": 1, "method": "C", "completed": True,
            "architecture": architecture, "seed": seed, "smoke": False,
            "data_revision": REVISION, "label_names": labels,
            "config": training_config(architecture), "parameter_count": 10,
            "sizes": {"train": EXPECTED_ROWS["train"], "validation": EXPECTED_ROWS["validation"]},
            "metrics": {**{name: 0.4 for name in METRIC_NAMES}, "macro_f1": macro_f1}}


def write_example_artifacts(folder, *, smoke=False):
    """Fixture: bytes không phải model thật, chỉ để kiểm ràng buộc artifact."""
    metadata = example_run("bert", 42, 0.4)
    metadata["smoke"] = smoke
    n = 32 if smoke else EXPECTED_ROWS["validation"]
    if smoke:
        metadata["sizes"] = {"train": 64, "validation": n}
    labels = metadata["label_names"]
    write_json(folder / "label_mapping.json",
               {"label_names": labels, "label2id": {name: i for i, name in enumerate(labels)}})
    write_json(folder / "checkpoint/config.json",
               {"model_type": "bert", "problem_type": "multi_label_classification",
                "id2label": {str(i): name for i, name in enumerate(labels)}})
    (folder / "checkpoint/model.safetensors").write_bytes(b"unit-test-only-not-a-real-model")
    np.savez(folder / "validation_scores.npz", ids=np.asarray([f"id_{i}" for i in range(n)]),
             scores=np.zeros((n, 28)), label_names=np.asarray(labels))
    write_json(folder / "validation_metrics.json",
               {**metadata["metrics"], "threshold": 0.5, "n_labels": 28, "n_samples": n})
    metadata["artifact_sha256"] = artifact_hashes(folder)
    write_json(folder / "run_metadata.json", metadata)
    return metadata


class TransformerTest(unittest.TestCase):
    def test_c1_defaults_and_smoke_are_separate(self):
        full = training_config("bert")
        self.assertEqual(full["checkpoint"], "google-bert/bert-base-cased")
        self.assertEqual(full["epochs"], 4)
        self.assertEqual(full["effective_batch_size"], 16)
        self.assertEqual(full["batch_size"], 16)
        self.assertEqual(full["gradient_accumulation"], 1)
        self.assertEqual(full["padding"], "dynamic_batch_trim")
        self.assertEqual(training_config("bert", smoke=True)["epochs"], 1)
        with self.assertRaises(ValueError):
            training_config("bert", gradient_accumulation=0)

    def test_positive_weights_use_positive_and_negative_counts(self):
        truth = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(positive_weights(truth), [3, 1, 4])
        with self.assertRaises(ValueError):
            positive_weights(np.array([[2]]))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Collate test cần torch")
    def test_dynamic_padding_trims_only_padding_and_keeps_28_targets(self):
        import torch
        from scripts.transformers.train_transformer import trim_padding_collate
        labels = torch.arange(28, dtype=torch.float32)
        items = [
            {"input_ids": torch.tensor([2, 3, 0, 0, 0, 0]),
             "attention_mask": torch.tensor([1, 1, 0, 0, 0, 0]),
             "token_type_ids": torch.tensor([0, 0, 0, 0, 0, 0]), "labels": labels},
            {"input_ids": torch.tensor([4, 5, 6, 0, 0, 0]),
             "attention_mask": torch.tensor([1, 1, 1, 0, 0, 0]),
             "token_type_ids": torch.tensor([0, 0, 0, 0, 0, 0]), "labels": labels + 1},
        ]
        batch = trim_padding_collate(items)
        self.assertEqual(batch["input_ids"].shape, (2, 3))
        self.assertEqual(batch["attention_mask"].shape, (2, 3))
        self.assertEqual(batch["token_type_ids"].shape, (2, 3))
        self.assertEqual(batch["labels"].shape, (2, 28))
        self.assertTrue(torch.equal(batch["labels"][1], labels + 1))
        self.assertEqual(batch["input_ids"][1].tolist(), [4, 5, 6])
        items[0]["attention_mask"] = torch.tensor([0, 0, 0, 0, 1, 1])
        with self.assertRaises(ValueError):
            trim_padding_collate(items)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch chưa cài; test loss cần torch")
    def test_loss_is_multilabel_bce_with_float_targets(self):
        import torch
        logits = torch.tensor([[0.0, 0.0], [2.0, -2.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        loss = multilabel_loss(logits, targets)
        expected = torch.nn.BCEWithLogitsLoss()(logits, targets)
        self.assertAlmostEqual(loss.item(), expected.item())
        loss.backward()
        self.assertEqual(logits.grad.shape, targets.shape)
        self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError):
            multilabel_loss(logits, targets.long())

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                         "Tiny-head test cần torch + transformers")
    def test_all_three_real_heads_support_28_logits_and_backward(self):
        """Model nhỏ khởi tạo ngẫu nhiên: kiểm kiến trúc, không phải kết quả đồ án."""
        import torch
        torch.set_num_threads(4)
        from transformers import (BertConfig, BertForSequenceClassification,
                                  RobertaConfig, RobertaForSequenceClassification,
                                  DistilBertConfig, DistilBertForSequenceClassification)
        common = dict(vocab_size=64, num_labels=28, problem_type="multi_label_classification")
        models = [
            BertForSequenceClassification(BertConfig(**common, hidden_size=16, num_hidden_layers=1,
                                                     num_attention_heads=2, intermediate_size=32,
                                                     max_position_embeddings=32)),
            RobertaForSequenceClassification(RobertaConfig(**common, hidden_size=16, num_hidden_layers=1,
                                                           num_attention_heads=2, intermediate_size=32,
                                                           max_position_embeddings=32)),
            DistilBertForSequenceClassification(DistilBertConfig(**common, dim=16, hidden_dim=32,
                                                               n_layers=1, n_heads=2,
                                                               max_position_embeddings=32)),
        ]
        ids = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
        targets = torch.zeros(2, 28, dtype=torch.float32)
        targets[0, [1, 17]] = 1  # Hai cảm xúc đồng thời trong cùng một mẫu.
        targets[1, 27] = 1
        for model in models:
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
            self.assertEqual(logits.shape, (2, 28))
            loss = multilabel_loss(logits, targets)
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in model.parameters()))

    def test_architecture_ranking_uses_mean_then_sample_std_then_cost(self):
        runs = {"bert": [example_run("bert", s, f) for s, f in zip([42, 123, 2026], [.2, .4, .6])],
                "roberta": [example_run("roberta", s, .4) for s in [42, 123, 2026]],
                "distilbert": [example_run("distilbert", s, .4) for s in [42, 123, 2026]]}
        for run in runs["distilbert"]:
            run["parameter_count"] = 5
        table = summarize_architectures(runs)
        self.assertEqual(table[0]["architecture"], "distilbert")
        bert = next(row for row in table if row["architecture"] == "bert")
        self.assertAlmostEqual(bert["metrics"]["macro_f1"]["std"], .2)

    def test_seed_summary_rejects_smoke_two_seeds_and_config_mixing(self):
        runs = {"bert": [example_run("bert", s, .4) for s in [42, 123, 2026]]}
        bad = copy.deepcopy(runs)
        bad["bert"][0]["smoke"] = True
        with self.assertRaises(ValueError):
            summarize_architectures(bad)
        with self.assertRaises(ValueError):
            summarize_architectures({"bert": runs["bert"][:2]})
        bad = copy.deepcopy(runs)
        bad["bert"][1]["config"]["learning_rate"] = 1e-3
        with self.assertRaises(ValueError):
            summarize_architectures(bad)

    def test_incomplete_resume_requests_explicit_restart_before_loading_models(self):
        from scripts.transformers import train_transformer
        from src.models.transformer import run_folder
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = run_folder(root, "bert", 42)
            write_json(folder / "run_metadata.json", {"completed": False})
            args = SimpleNamespace(architecture="bert", seed=42, smoke=False, weighted=False,
                                   epochs=None, learning_rate=None, batch_size=16,
                                   gradient_accumulation=1, max_length=128, device="cpu",
                                   revision="main", resume=True, overwrite=False)
            with patch.object(train_transformer, "ROOT", root), patch.object(
                    train_transformer, "parse_args", return_value=args):
                with self.assertRaisesRegex(RuntimeError, "--overwrite"):
                    train_transformer.main()
    def test_run_integrity_checks_smoke_mapping_and_changed_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_example_artifacts(folder, smoke=True)
            load_transformer_run(folder)
            with self.assertRaises(ValueError):
                load_transformer_run(folder, require_full=True)
            (folder / "checkpoint/model.safetensors").write_bytes(b"replaced-model")
            with self.assertRaises(ValueError):
                load_transformer_run(folder)

    def test_demo_selection_and_thresholds_are_bound_to_same_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "run"
            metadata = write_example_artifacts(folder)
            selection = {"artifact_version": 1, "method": "C", "smoke": False,
                         "run_dir": "run", "architecture": "bert", "seed": 42,
                         "run_metadata_sha256": sha256(folder / "run_metadata.json")}
            path = root / "selected.json"
            write_json(path, selection)
            load_demo_selection(root, path)
            thresholds = root / "thresholds.json"
            write_json(thresholds, {"label_names": metadata["label_names"],
                                    "run_metadata_sha256": "other-run", "thresholds": [0.3] * 28})
            with self.assertRaises(ValueError):
                load_demo_selection(root, path, thresholds_path=thresholds)


if __name__ == "__main__":
    unittest.main()
