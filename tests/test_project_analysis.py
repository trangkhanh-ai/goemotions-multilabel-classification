"""Bảng đồ án dùng metric GIẢ; không mở dataset và không tải model."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.analysis.export_project_analysis import (aggregate_c, collect_training_costs, export_analysis,
                                             export_figures, frozen_protocol, make_record,
                                             rare_comparisons, write_analysis_markdown)
from src.datasets.goemotions import EXPECTED_ROWS, REVISION, sha256
from src.evaluation.protocols import save_json
from src.models.transformer import ARCHITECTURES, run_folder


def synthetic_metric(labels, split, f1):
    return {"n_labels": len(labels), "n_samples": EXPECTED_ROWS[split],
            "macro_f1": f1, "micro_f1": f1,
            "per_label": [{"label_id": j, "label": name, "support": 2, "tp": 1, "fp": 1, "fn": 1,
                           "precision": f1, "recall": f1, "f1": f1} for j, name in enumerate(labels)]}


def synthetic_records(labels, seeds=(42, 123, 2026)):
    records = []
    for split in ("validation", "test"):
        for variant, mode, value in (("standard", "fixed", 0.6), ("balanced", "tuned", 0.4)):
            records.append(make_record(labels, split, "A", variant, "tfidf_lr", None, mode,
                                       synthetic_metric(labels, split, value), "synthetic"))
        for architecture in ARCHITECTURES:
            for i, seed in enumerate(seeds):
                for mode, value in (("fixed", 0.5 + i * 0.1), ("tuned", 0.4 + i * 0.1)):
                    records.append(make_record(labels, split, "C", "standard", architecture, seed, mode,
                                               synthetic_metric(labels, split, value), "synthetic", config="same"))
    return records


class ProjectAnalysisTest(unittest.TestCase):
    def test_mean_sample_std_uses_all_paired_seeds(self):
        records = synthetic_records(["a", "b"])
        result, group = aggregate_c(records, "bert", "test", "fixed", [42, 123, 2026])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["ddof"], 1)
        self.assertAlmostEqual(result["macro_f1_mean"], 0.6)
        self.assertAlmostEqual(result["macro_f1_std"], 0.1)
        self.assertEqual([row["seed"] for row in group], [42, 123, 2026])

    def test_missing_seed_stays_missing_instead_of_zero_or_two_seed_mean(self):
        records = synthetic_records(["a", "b"])
        records = [row for row in records if not (row["method"] == "C" and row["architecture"] == "bert"
                                                 and row["seed"] == 2026)]
        result, group = aggregate_c(records, "bert", "test", "fixed", [42, 123, 2026])
        self.assertEqual(result["status"], "missing")
        self.assertIsNone(result["macro_f1_mean"])
        self.assertEqual(group, [])

    def test_different_configs_cannot_be_averaged_as_seed_variance(self):
        records = synthetic_records(["a", "b"])
        for row in records:
            if row["method"] == "C" and row["architecture"] == "bert" and row["seed"] == 2026:
                row["config_signature"] = "changed batch-size"
        result, _ = aggregate_c(records, "bert", "test", "fixed", [42, 123, 2026])
        self.assertEqual(result["status"], "missing")
        self.assertIsNone(result["micro_f1_mean"])

    def test_rare_labels_come_from_train_and_negative_deltas_are_kept(self):
        labels = [f"label_{j}" for j in range(7)]
        records = synthetic_records(labels)
        support = [100, 5, 4, 3, 2, 1, 101]
        rows = rare_comparisons(records, labels, support, [42, 123, 2026])
        a_rows = [row for row in rows if row["method"] == "A" and row["split"] == "test"]
        self.assertEqual([row["label"] for row in a_rows], ["label_5", "label_4", "label_3", "label_2", "label_1"])
        self.assertTrue(all(row["delta_f1_mean"] < 0 for row in a_rows))
        c_row = next(row for row in rows if row["method"] == "C" and row["architecture"] == "bert" and row["split"] == "test")
        self.assertAlmostEqual(c_row["delta_f1_mean"], -0.1)
        self.assertAlmostEqual(c_row["delta_f1_std"], 0.0)
        val_row = next(row for row in rows if row["method"] == "C" and row["split"] == "validation")
        self.assertTrue(val_row["validation_calibrated"])

    def test_missing_train_support_does_not_select_rare_labels_using_test(self):
        rows = rare_comparisons(synthetic_records(["a", "b"]), ["a", "b"], None, [42, 123, 2026])
        self.assertEqual(rows[0]["status"], "missing")
        self.assertNotIn("label", rows[0])

    def test_frozen_protocol_rejects_changed_validation_score_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            labels = ["a", "b"]
            metadata = {"method": "C", "checkpoint": "model", "model_revision": "a" * 40,
                        "architecture": "bert", "seed": 42}
            save_json(folder / "run_metadata.json", metadata)
            (folder / "validation_scores.npz").write_bytes(b"scores")
            protocol = {"protocol_version": 1, "smoke": False, "label_names": labels,
                        "data_revision": REVISION, "selection_data": "validation", "frozen_at_utc": "2026-10-08",
                        "checkpoint": "model", "model_revision": "a" * 40, "architecture": "bert", "seed": 42,
                        "run_metadata_sha256": sha256(folder / "run_metadata.json"), "validation_scores_sha256": "wrong",
                        "configurations": [{"threshold_mode": mode, "thresholds": 0.5} for mode in ("fixed", "global", "tuned")]}
            save_json(folder / "final_protocol.json", protocol)
            issues = []
            self.assertIsNone(frozen_protocol(folder, metadata, labels, issues))
            self.assertEqual(issues[0]["status"], "invalid")

    def test_invalid_per_label_mapping_is_not_exported_as_measured_f1(self):
        labels = ["a", "b"]
        metric = synthetic_metric(labels, "test", 0.5)
        metric["per_label"][0]["label"] = "different"
        record = make_record(labels, "test", "C", "standard", "bert", 42, "fixed", metric, "synthetic")
        self.assertEqual(record["status"], "invalid")
        self.assertIsNone(record["metric"])

    def test_missing_everything_exports_blank_values_and_an_explicit_missing_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            save_json(root / "data/labels.json", ["a", "b"])
            result = export_analysis(root, figures=False)
            self.assertEqual(result["n_configurations_ok"], 0)
            output = root / "reports/project_results"
            per_label = pd.read_csv(output / "per_label.csv")
            self.assertTrue((per_label["status"] == "missing").all())
            self.assertTrue(per_label["f1"].isna().all())
            aggregate = pd.read_csv(output / "aggregate_metrics.csv")
            self.assertTrue((aggregate["status"] == "missing").all())
            self.assertTrue(aggregate["macro_f1_mean"].isna().all())
            self.assertTrue((output / "missing_artifacts.csv").is_file())
            self.assertTrue((output / "learning_curves.csv").is_file())

    def test_complete_synthetic_metrics_render_three_figures_without_any_dataset(self):
        seeds = [42, 123, 2026]
        records = synthetic_records(["a", "b"])
        aggregates = [aggregate_c(records, architecture, split, mode, seeds)[0]
                      for split in ("validation", "test") for architecture in ARCHITECTURES
                      for mode in ("fixed", "tuned")]
        curves = [{"architecture": architecture, "seed": seed, "epoch": epoch, "status": "ok",
                   "validation_macro_f1": 0.4 + epoch * 0.1,
                   "validation_micro_f1": 0.5 + epoch * 0.1}
                  for architecture in ARCHITECTURES for seed in seeds for epoch in (1, 2)]
        with tempfile.TemporaryDirectory() as temporary:
            issues = []
            paths = export_figures(Path(temporary), curves, aggregates, seeds, issues)
            self.assertEqual(len(paths), 3)
            self.assertEqual(issues, [])
            self.assertTrue(all(Path(path).stat().st_size > 1000 for path in paths))

    def test_human_analysis_uses_test_only_when_all_four_models_have_paired_rare_results(self):
        labels = [f"label_{j}" for j in range(6)]
        seeds = [42, 123, 2026]
        support = [1, 2, 3, 4, 5, 100]
        records = synthetic_records(labels)
        rare = rare_comparisons(records, labels, support, seeds)
        costs = [{"architecture": architecture, "status": "missing", "n_seeds_available": 0,
                  "n_seeds_expected": 3} for architecture in ARCHITECTURES]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            selected = write_analysis_markdown(output, records, rare, labels, support, costs, [])
            self.assertEqual(selected["rare_primary_split"], "test")
            text = (output / "ANALYSIS.md").read_text(encoding="utf-8")
            self.assertIn("Bảng chính: TEST", text)
            self.assertIn("-0.1000", text)
            self.assertIn("-0.2000", text)
            self.assertIn("C1: BERT", text)
            self.assertIn("Train + | Validation + | Test +", text)
            self.assertIn("Thiếu run/metadata", text)
            records = [row for row in records if not (row["method"] == "C" and row["architecture"] == "bert"
                                                      and row["seed"] == 2026 and row["split"] == "test"
                                                      and row["threshold_mode"] == "tuned")]
            rare = rare_comparisons(records, labels, support, seeds)
            selected = write_analysis_markdown(output, records, rare, labels, support, costs, [])
            self.assertEqual(selected["rare_primary_split"], "validation")
            self.assertFalse(selected["rare_test_complete"])
            self.assertIn("Bảng tạm: VALIDATION", (output / "ANALYSIS.md").read_text(encoding="utf-8"))

    def test_training_costs_use_actual_full_metadata_and_do_not_invent_partial_sum(self):
        labels, seeds = ["a", "b"], [42, 123, 2026]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for architecture in ARCHITECTURES:
                for index, seed in enumerate(seeds):
                    metadata = {"method": "C", "architecture": architecture, "seed": seed, "completed": True,
                                "smoke": False, "data_revision": REVISION, "label_names": labels,
                                "sizes": {"train": EXPECTED_ROWS["train"], "validation": EXPECTED_ROWS["validation"]},
                                "elapsed_seconds": 10 * (index + 1), "parameter_count": 1000}
                    save_json(run_folder(root, architecture, seed) / "run_metadata.json", metadata)
            rows = collect_training_costs(root, labels, seeds, [])
            self.assertTrue(all(row["status"] == "ok" for row in rows))
            self.assertEqual(rows[0]["elapsed_seconds_sum"], 60)
            self.assertEqual(rows[0]["elapsed_seconds_mean"], 20)
            self.assertEqual(rows[0]["parameter_count"], 1000)
            (run_folder(root, "bert", 2026) / "run_metadata.json").unlink()
            row = collect_training_costs(root, labels, seeds, [])[0]
            self.assertEqual(row["status"], "missing")
            self.assertEqual(row["n_seeds_available"], 2)
            self.assertIsNone(row["elapsed_seconds_sum"])
            self.assertIsNone(row["elapsed_seconds_mean"])


if __name__ == "__main__":
    unittest.main()
