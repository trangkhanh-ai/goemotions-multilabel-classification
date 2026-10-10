import unittest
import numpy as np
from src.learning.baseline_numpy import simple_tfidf, fit_multilabel_logistic, sigmoid


class EducationalTests(unittest.TestCase):
    def test_tfidf_rare_word_has_larger_idf_and_rows_normalized(self):
        x, words, idf = simple_tfidf(["common rare", "common common"])
        self.assertGreater(idf[words.index("rare")], idf[words.index("common")])
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), [1, 1])

    def test_binary_heads_can_predict_multiple_labels(self):
        x = np.eye(3)
        y = np.array([[1, 1], [0, 1], [1, 0]])
        w, b, losses = fit_multilabel_logistic(x, y, steps=800)
        self.assertLess(losses[-1], losses[0])
        np.testing.assert_array_equal(sigmoid(x @ w + b) >= 0.5, y)


if __name__ == "__main__":
    unittest.main()
