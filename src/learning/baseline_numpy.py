"""Ví dụ NumPy để học TF-IDF và Logistic Regression đa nhãn.

Đây là minh họa trên dữ liệu nhỏ, không thay kết quả baseline scikit-learn.
"""
import re
import numpy as np


def simple_tfidf(texts):
    tokens = [re.findall(r"\b\w+\b", text.lower()) for text in texts]
    vocabulary = sorted({word for sentence in tokens for word in sentence})
    if not texts or not vocabulary:
        raise ValueError("Ví dụ cần có văn bản và từ vựng")
    counts = np.array([[sentence.count(word) for word in vocabulary] for sentence in tokens], dtype=float)
    document_frequency = (counts > 0).sum(axis=0)
    idf = np.log((1 + len(texts)) / (1 + document_frequency)) + 1
    features = counts * idf
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)
    return features, vocabulary, idf


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -500, 500)))


def fit_multilabel_logistic(features, labels, *, steps=500, learning_rate=0.5):
    """Một cột W là một LR nhị phân; gradient cập nhật cùng lúc các nhãn."""
    x, y = np.asarray(features, dtype=float), np.asarray(labels, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError("X là N×D, Y là N×L")
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("X hữu hạn, Y multi-hot 0/1")
    weights = np.zeros((x.shape[1], y.shape[1]))
    bias = np.zeros(y.shape[1])
    losses = []
    for _ in range(steps):
        logits = x @ weights + bias
        losses.append(float(np.mean(np.logaddexp(0, logits) - y * logits)))
        error = (sigmoid(logits) - y) / y.size
        weights -= learning_rate * (x.T @ error)
        bias -= learning_rate * error.sum(axis=0)
    return weights, bias, losses
