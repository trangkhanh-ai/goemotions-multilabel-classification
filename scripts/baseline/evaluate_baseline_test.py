"""Đánh giá toàn bộ cấu hình A đã khóa trước test.

Sau khi nhóm chốt: python -m scripts.baseline.freeze_baseline
                  python -m scripts.baseline.evaluate_baseline_test
Chạy lại với cùng protocol sẽ trả kết quả đã có; không dùng test chọn cấu hình.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.baseline import load_run_metadata
from src.datasets.goemotions import REVISION, load_goemotions, multi_hot, sha256
from src.evaluation.metrics import evaluate_multilabel, validate_thresholds

from src.paths import ROOT


def evaluate_frozen_protocol(root):
    protocol_path = root / "data/processed/baseline/final_protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError("Chưa khóa protocol. Nhóm chốt trước, rồi chạy scripts.baseline.freeze_baseline.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    fingerprint = sha256(protocol_path)
    labels = json.loads((root / "data/labels.json").read_text(encoding="utf-8"))
    if (protocol["schema_version"] != 1 or protocol["data_revision"] != REVISION
            or protocol["label_names"] != labels
            or set(protocol["runs"]) != {"standard", "balanced"}):
        raise ValueError("Protocol không khớp revision hoặc thứ tự nhãn")
    # Kiểm toàn bộ model/ngưỡng trước khi mở nhãn test.
    for variant, run in protocol["runs"].items():
        folder = root / run["folder"]
        metadata = load_run_metadata(folder, variant, labels)
        if (metadata["artifact_sha256"] != run["artifact_sha256"]
                or sha256(folder / "thresholds_validation.json") != run["threshold_file_sha256"]):
            raise ValueError("Model/scores/ngưỡng đã đổi sau khi khóa protocol")
    names = set()
    for config in protocol["configurations"]:
        expected_name = f"{config['variant']}_{config['threshold_mode']}"
        if (config["variant"] not in protocol["runs"] or config["name"] != expected_name
                or config["threshold_mode"] not in ("fixed", "global", "tuned")
                or config["name"] in names):
            raise ValueError("Danh sách cấu hình trong protocol không hợp lệ")
        names.add(config["name"])
        validate_thresholds(config["thresholds"], len(labels))
    if names != {f"{variant}_{mode}" for variant in ("standard", "balanced")
                 for mode in ("fixed", "global", "tuned")}:
        raise ValueError("Protocol phải giữ đủ baseline gốc và các cấu hình so sánh đã khóa")

    output = root / "data/processed/baseline/final"
    summary_path = output / "final_results.json"
    marker_path = output / "protocol.sha256"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["protocol_sha256"] != fingerprint:
            raise ValueError("Đã có kết quả test thuộc protocol khác; không chọn lại bằng test")
        return summary  # Tái sử dụng kết quả, không đọc lại test.
    if marker_path.exists() and marker_path.read_text().strip() != fingerprint:
        raise ValueError("Lần chạy test trước dùng protocol khác")
    output.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(fingerprint, encoding="ascii")

    frames, test_labels, _ = load_goemotions(root, write_metadata=False, splits=("test",))
    if test_labels != labels:
        raise ValueError("Nhãn test không khớp mapping đã huấn luyện")
    test = frames["test"]
    truth = multi_hot(test["labels"].tolist(), len(labels))
    scores_by_variant = {}
    for variant, run in protocol["runs"].items():
        model = joblib.load(root / run["folder"] / "model.joblib")
        scores = model.predict_proba(test["text"].tolist())
        scores_by_variant[variant] = scores
        np.savez_compressed(output / f"{variant}_test_scores.npz",
                            ids=test["id"].to_numpy(dtype=str), scores=scores,
                            label_names=np.asarray(labels, dtype=str))

    results = []
    for config in protocol["configurations"]:
        metrics = evaluate_multilabel(truth, scores_by_variant[config["variant"]], labels,
                                     threshold=config["thresholds"])
        record = {"name": config["name"], "split": "test", "protocol_sha256": fingerprint,
                  "data_revision": REVISION, "variant": config["variant"],
                  "threshold_mode": config["threshold_mode"], "metrics": metrics}
        (output / f"{config['name']}_test_metrics.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results.append(record)
    # Bảng nhãn hiếm giữ cả kết quả tăng và giảm, so với baseline gốc standard_fixed.
    original = next(row for row in results if row["name"] == "standard_fixed")["metrics"]
    rare_rows = []
    for label_id in protocol["rare_label_ids_from_train"]:
        for result in results:
            item = result["metrics"]["per_label"][label_id]
            rare_rows.append({"label": labels[label_id], "configuration": result["name"],
                              "test_support": item["support"], "f1": item["f1"],
                              "f1_change_vs_standard_fixed":
                              item["f1"] - original["per_label"][label_id]["f1"]})
    pd.DataFrame(rare_rows).to_csv(output / "rare_labels_test.csv", index=False, encoding="utf-8-sig")
    summary = {"protocol_sha256": fingerprint, "data_revision": REVISION,
               "selected_on_validation": protocol["selected_configuration"], "results": results}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    argparse.ArgumentParser(description="Evaluate the baseline configurations in the frozen protocol").parse_args()
    summary = evaluate_frozen_protocol(ROOT)
    print("Final baseline test (all configurations were frozen before reading test):")
    for row in summary["results"]:
        metrics = row["metrics"]
        print(f"{row['name']:18s} Macro-F1={metrics['macro_f1']:.4f} "
              f"Micro-F1={metrics['micro_f1']:.4f}")
    print(f"Selection made on validation: {summary['selected_on_validation']}")
    print("Artifacts: data/processed/baseline/final/")


if __name__ == "__main__":
    main()
