"""Shared multi-label metrics for the 28 GoEmotions labels."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import hamming_loss, precision_recall_fscore_support


def validate_multilabel_inputs(y_true, scores):
    """Chặn dữ liệu sai trước khi tính metric hoặc tìm ngưỡng."""
    truth = np.asarray(y_true)
    probabilities = np.asarray(scores)
    if truth.ndim != 2 or probabilities.shape != truth.shape:
        raise ValueError("y_true và scores phải cùng kích thước N × số nhãn")
    if not truth.shape[0] or not truth.shape[1]:
        raise ValueError("Không đánh giá ma trận rỗng")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("y_true chỉ được chứa 0 và 1")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("scores phải hữu hạn và nằm trong [0, 1]")
    return truth, probabilities


def validate_thresholds(threshold, n_labels):
    """Một ngưỡng chung hoặc đúng một ngưỡng cho mỗi nhãn."""
    limits = np.asarray(threshold, dtype=float)
    if limits.ndim > 1 or (limits.ndim == 1 and limits.shape != (n_labels,)):
        raise ValueError("threshold phải là một số hoặc một giá trị cho mỗi nhãn")
    if not np.isfinite(limits).all() or np.any((limits < 0) | (limits > 1)):
        raise ValueError("threshold phải hữu hạn và nằm trong [0, 1]")
    return limits


def evaluate_multilabel(
    y_true: np.ndarray,
    scores: np.ndarray,
    label_names: list[str],
    threshold: float | list[float] | np.ndarray = 0.5,
) -> dict:
    """Evaluate scores in the canonical label order; never choose a threshold here."""
    truth, probabilities = validate_multilabel_inputs(y_true, scores)
    if len(label_names) != truth.shape[1] or len(set(label_names)) != len(label_names):
        raise ValueError("Số tên nhãn không khớp số cột của ma trận")
    limits = validate_thresholds(threshold, truth.shape[1])

    predictions = (probabilities >= limits).astype(np.uint8)
    positive = truth.astype(bool)
    predicted = predictions.astype(bool)
    tp = (positive & predicted).sum(axis=0)
    fp = (~positive & predicted).sum(axis=0)
    fn = (positive & ~predicted).sum(axis=0)
    tn = (~positive & ~predicted).sum(axis=0)
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        truth, predictions, average=None, zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        truth, predictions, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        truth, predictions, average="macro", zero_division=0
    )
    return {
        "n_samples": int(truth.shape[0]),
        "n_labels": int(truth.shape[1]),
        "threshold": float(limits) if limits.ndim == 0 else limits.tolist(),
        "zero_division": 0,
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "hamming_loss": float(hamming_loss(truth, predictions)),
        "empty_prediction_count": int((predictions.sum(axis=1) == 0).sum()),
        "mean_predicted_labels": float(predictions.sum(axis=1).mean()),
        "per_label": [
            {
                "label_id": i,
                "label": name,
                "support": int(support[i]),
                "predicted_support": int(tp[i] + fp[i]),
                "tp": int(tp[i]), "fp": int(fp[i]),
                "fn": int(fn[i]), "tn": int(tn[i]),
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f1[i]),
            }
            for i, name in enumerate(label_names)
        ],
    }
