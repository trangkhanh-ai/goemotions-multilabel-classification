"""Các hàm nhỏ dùng chung cho phân tích baseline đa nhãn."""

from __future__ import annotations

import json
import numpy as np
from sklearn.metrics import f1_score

from src.data import REVISION, sha256
from src.metrics import validate_multilabel_inputs, validate_thresholds

THRESHOLD_GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)


def load_aligned_scores(path, expected_ids, expected_labels):
    """Đọc điểm dự đoán và ghép theo ID, không dựa vào thứ tự dòng của file."""
    with np.load(path, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        scores = saved["scores"].astype(float)
        labels = saved["label_names"].astype(str).tolist()
    wanted_ids = np.asarray(expected_ids, dtype=str)
    if ids.ndim != 1 or wanted_ids.ndim != 1:
        raise ValueError("ID phải là vector một chiều")
    if labels != list(expected_labels):
        raise ValueError("Thứ tự 28 nhãn của file điểm không khớp dữ liệu")
    if scores.shape != (len(ids), len(labels)):
        raise ValueError("Ma trận điểm không có kích thước N × số nhãn")
    if len(set(ids)) != len(ids) or len(set(wanted_ids)) != len(wanted_ids):
        raise ValueError("ID bị trùng, không thể ghép an toàn")
    position = {sample_id: i for i, sample_id in enumerate(ids)}
    if set(position) != set(wanted_ids):
        raise ValueError("Tập ID trong file điểm khác tập ID của split cần đánh giá")
    aligned = scores[[position[sample_id] for sample_id in wanted_ids]]
    if not np.isfinite(aligned).all() or np.any((aligned < 0) | (aligned > 1)):
        raise ValueError("Điểm dự đoán phải hữu hạn trong [0, 1]")
    return aligned


def tune_thresholds(y_true, scores):
    """Chọn ngưỡng F1 cho từng nhãn trên validation, không xem test.

    Lưới cố định 0,05..0,95; khi hòa chọn ngưỡng gần 0,5 nhất. Đây là
    ước lượng trên cùng validation dùng để chọn ngưỡng, nên có thể lạc quan.
    """
    truth, probabilities = validate_multilabel_inputs(y_true, scores)
    chosen = []
    for label_id in range(truth.shape[1]):
        candidates = []
        for threshold in THRESHOLD_GRID:
            predicted = probabilities[:, label_id] >= threshold
            score = f1_score(truth[:, label_id], predicted, zero_division=0)
            candidates.append((score, -round(abs(threshold - 0.5), 10), threshold))
        chosen.append(float(max(candidates)[2]))
    return np.asarray(chosen)


def tune_global_threshold(y_true, scores):
    """Chọn một ngưỡng chung bằng Macro-F1; lưu toàn bộ đường cong để kiểm tra."""
    truth, probabilities = validate_multilabel_inputs(y_true, scores)
    curve = []
    for threshold in THRESHOLD_GRID:
        predicted = probabilities >= threshold
        curve.append({"threshold": float(threshold),
                      "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
                      "micro_f1": float(f1_score(truth, predicted, average="micro", zero_division=0)),
                      "hamming_loss": float(np.mean(truth != predicted))})
    best = max(curve, key=lambda row: (
        row["macro_f1"], -round(abs(row["threshold"] - 0.5), 10), row["threshold"]
    ))
    return best["threshold"], curve


def load_run_metadata(folder, variant, labels):
    """Kiểm model và scores thực sự thuộc cùng một lần chạy full."""
    saved = json.loads((folder / "validation_metrics.json").read_text(encoding="utf-8"))
    if saved.get("artifact_version") != 2:
        raise ValueError("Artifacts cũ chưa có hash. Chạy lại scripts.run_baseline.")
    if (saved["variant"] != variant or saved["data_revision"] != REVISION
            or saved["smoke"] or saved["label_names"] != list(labels)):
        raise ValueError("Variant/revision/nhãn của model không khớp bản full cần dùng")
    for filename, expected_hash in saved["artifact_sha256"].items():
        if sha256(folder / filename) != expected_hash:
            raise ValueError(f"{filename} đã thay đổi hoặc bị trộn từ lần chạy khác")
    return saved


def load_thresholds(folder, metadata, mode):
    """Ngưỡng được ràng buộc với hash model, scores, revision và thứ tự nhãn."""
    if mode == "fixed":
        return np.asarray(0.5)
    if mode not in ("global", "tuned"):
        raise ValueError("mode phải là fixed, global hoặc tuned")
    saved = json.loads((folder / "thresholds_validation.json").read_text(encoding="utf-8"))
    for key in ("variant", "data_revision", "label_names", "artifact_sha256"):
        if saved.get(key) != metadata[key]:
            raise ValueError("Ngưỡng đã cũ hoặc thuộc model khác. Chạy lại scripts.analyze_baseline.")
    value = saved["global_threshold"] if mode == "global" else saved["thresholds"]
    return validate_thresholds(value, len(metadata["label_names"]))


def label_error_pairs(y_true, scores, label_names, threshold=0.5):
    """Đếm FN nhãn A và FP nhãn B cùng câu; không ép đa nhãn thành một lớp."""
    truth, probabilities = validate_multilabel_inputs(y_true, scores)
    limits = validate_thresholds(threshold, truth.shape[1])
    predicted = probabilities >= limits
    missed = truth.astype(bool) & ~predicted
    extra = ~truth.astype(bool) & predicted
    counts = missed.astype(np.int64).T @ extra.astype(np.int64)
    rows = []
    for a, b in np.argwhere(counts > 0):
        rows.append({"missed_label": label_names[a], "extra_label": label_names[b],
                     "count": int(counts[a, b]), "missed_label_total_fn": int(missed[:, a].sum()),
                     "fraction_of_fn": float(counts[a, b] / missed[:, a].sum()),
                     "example_row": int(np.flatnonzero(missed[:, a] & extra[:, b])[0])})
    return sorted(rows, key=lambda row: (-row["count"], row["missed_label"], row["extra_label"]))
