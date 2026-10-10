"""Minh họa từng bước NumPy, không phải thí nghiệm trên GoEmotions."""
import numpy as np
from src.learning.baseline_numpy import simple_tfidf, fit_multilabel_logistic, sigmoid


def main():
    texts = ["thank you happy", "thank you", "sad alone", "sad", "happy wonderful", "happy thank you"]
    labels = ["gratitude", "joy", "sadness"]
    y = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 1, 0]])
    x, vocabulary, idf = simple_tfidf(texts)
    weights, bias, losses = fit_multilabel_logistic(x, y)
    scores = sigmoid(x @ weights + bias)
    print("DỮ LIỆU MINH HỌA TỰ TẠO; không dùng số này làm kết quả GoEmotions.")
    print("Vocabulary:", vocabulary)
    print("IDF:", idf.round(3))
    print("TF-IDF câu 1:", x[0].round(3))
    print("BCE đầu/cuối:", round(losses[0], 4), round(losses[-1], 4))
    for text, row in zip(texts, scores):
        print(text, "->", row.round(3), "->", [name for name, score in zip(labels, row) if score >= 0.5])


if __name__ == "__main__":
    main()
