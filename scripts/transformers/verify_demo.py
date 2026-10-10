"""Kiểm mô hình demo C thật với scores đã lưu, không huấn luyện hay chọn lại.

Chạy sau khi đã đủ C: python -m scripts.transformers.verify_demo --device cuda
Ảnh giao diện và kiểm HTTP được ghi riêng; script này kiểm hàm suy luận.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app import TransformerDemo
from src.models.baseline import load_aligned_scores
from src.datasets.goemotions import load_goemotions, sha256
from src.evaluation.protocols import read_json, save_json
from src.models.transformer import configure_console

from src.paths import ROOT


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    selection_path = ROOT / "data/processed/transformers/selected_model.json"
    selection = read_json(selection_path)
    predictor = TransformerDemo(selection_path, device=args.device)
    folder = ROOT / selection["run_dir"]
    frames, labels, _ = load_goemotions(ROOT, write_metadata=False, splits=("validation",))
    frame = frames["validation"]
    saved_scores = load_aligned_scores(folder / "validation_scores.npz", frame["id"], labels)
    thresholds = np.broadcast_to(np.asarray(predictor.thresholds), (28,))

    # Chọn ba câu validation không sát ngưỡng để kiểm cùng quyết định nhãn.
    # Batch/padding và FP16 có thể làm scores lệch nhẹ; không chọn model bằng phép kiểm này.
    indices = [i for i, text in enumerate(frame["text"])
               if text == text.strip() and np.min(np.abs(saved_scores[i] - thresholds)) > 0.02][:3]
    if len(indices) != 3:
        raise ValueError("Chưa có đủ ba câu cách ngưỡng để đối chiếu demo")
    cases = []
    for i in indices:
        message, rows = predictor.predict(frame.iloc[i]["text"])
        by_label = {row[0]: float(row[1]) for row in rows}
        scores = np.array([by_label[label] for label in labels])
        difference = float(np.max(np.abs(scores - saved_scores[i])))
        if difference > 0.005:
            raise ValueError(f"Demo lệch scores đã lưu quá 0,005 ở ID {frame.iloc[i]['id']}")
        selected = [label for label, score, limit in zip(labels, scores, thresholds) if score >= limit]
        expected = [label for label, score, limit in zip(labels, saved_scores[i], thresholds) if score >= limit]
        if selected != expected:
            raise ValueError("Demo và script đánh giá khác quyết định nhãn")
        cases.append({"id": frame.iloc[i]["id"], "text": frame.iloc[i]["text"],
                      "true_labels": [labels[j] for j in frame.iloc[i]["labels"]],
                      "predicted_labels": selected, "max_score_difference": difference,
                      "display": message})

    empty_message, empty_rows = predictor.predict("   ")
    long_message, long_rows = predictor.predict("a" * 20001)
    truncated_message, truncated_rows = predictor.predict("The meeting starts at nine tomorrow. " * 200)
    checks = {"empty_input": not empty_rows and "nhập" in empty_message,
              "overlong_input": not long_rows and "quá dài" in long_message,
              "truncation_notice": len(truncated_rows) == 28 and "được cắt" in truncated_message}
    if not all(checks.values()):
        raise ValueError(f"Luồng nhập demo chưa đúng: {checks}")
    save_json(ROOT / "reports/demo_verification.json", {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_inference_status": "PASS", "interface_status": "NOT_CHECKED_YET",
        "architecture": predictor.metadata["architecture"], "seed": predictor.metadata["seed"],
        "checkpoint_revision": predictor.metadata["model_revision"],
        "selection_sha256": sha256(selection_path), "thresholds": predictor.thresholds,
        "device": str(predictor.device), "score_tolerance": 0.005,
        "validation_cases": cases, "input_checks": checks,
        "note": "Sai số cho phép gồm FP16/padding và làm tròn bảng 4 chữ số. Không dùng test hoặc chọn lại model."})
    print(f"Demo inference PASS: {predictor.metadata['architecture']} seed {predictor.metadata['seed']}; "
          "ba câu đối chiếu và ba luồng nhập. Giao diện cần kiểm riêng.")


if __name__ == "__main__":
    main()
