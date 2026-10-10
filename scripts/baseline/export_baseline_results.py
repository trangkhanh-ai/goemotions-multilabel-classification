"""Xuất các bảng validation nhỏ để nhóm đọc trên GitHub; không mở test.

Chạy sau analyze: python -m scripts.baseline.export_baseline_results
Không xuất model, raw data hoặc ma trận scores lớn.
"""

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from src.models.baseline import load_run_metadata, load_thresholds
from src.datasets.goemotions import REVISION

from src.paths import ROOT


def export_results(root):
    labels = json.loads((root / "data/labels.json").read_text(encoding="utf-8"))
    destination = root / "reports/baseline_validation"
    rows = []
    for variant in ("standard", "balanced"):
        source = root / "data/processed/baseline"
        if variant == "balanced":
            source /= "balanced"
        source /= "full"
        metadata = load_run_metadata(source, variant, labels)
        analysis = json.loads((source / "analysis_validation.json").read_text(encoding="utf-8"))
        if analysis.get("artifact_sha256") != metadata["artifact_sha256"]:
            raise ValueError("Analysis is stale. Run scripts.baseline.analyze_baseline first.")
        # Đọc ngưỡng kiểm hash trước khi xuất; số liệu luôn thuộc đúng model.
        for mode, key in (("fixed", "fixed_0_5"), ("global", "global_on_validation"),
                          ("tuned", "tuned_on_validation")):
            threshold = load_thresholds(source, metadata, mode)
            metrics = analysis[key]
            row = {"configuration": f"{variant}_{mode}", "split": "validation",
                   "n_train": metadata["n_train"], "n_validation": metadata["n_validation"],
                   "threshold": "per_label" if mode == "tuned" else float(threshold)}
            for name in ("macro_f1", "micro_f1", "micro_precision", "micro_recall",
                         "macro_precision", "macro_recall", "hamming_loss"):
                row[name] = metrics[name]
            rows.append(row)
        target = destination / variant
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("validation_metrics.json", "thresholds_validation.json",
                         "per_label_validation.csv", "label_error_pairs_validation.csv",
                         "threshold_curve_validation.csv"):
            shutil.copyfile(source / filename, target / filename)
    pd.DataFrame(rows).to_csv(destination / "comparison.csv", index=False, encoding="utf-8-sig")
    (destination / "summary.json").write_text(json.dumps({
        "data_revision": REVISION, "label_names": labels,
        "note": "Validation only. Tuned rows use the same validation for threshold selection and scoring. Test not evaluated.",
        "configurations": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main():
    argparse.ArgumentParser(description="Export small, verified baseline validation tables for GitHub").parse_args()
    rows = export_results(ROOT)
    print(f"Exported {len(rows)} validation configurations to reports/baseline_validation/")
    print("Test not opened; models and prediction matrices are kept in data/processed/.")


if __name__ == "__main__":
    main()
