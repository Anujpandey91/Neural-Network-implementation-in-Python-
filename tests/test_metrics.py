import numpy as np

from mydl.metrics import (
    accuracy,
    confusion_matrix,
    precision,
    recall,
    f1_score,
)


class TestAccuracy:

    def test_perfect_accuracy(self):
        y_true = np.array([[1, 0, 1, 0]])
        y_pred = np.array([[1, 0, 1, 0]])

        assert accuracy(y_true, y_pred) == 1.0

    def test_partial_accuracy(self):
        y_true = np.array([[1, 0, 1, 0]])
        y_pred = np.array([[1, 1, 1, 0]])

        assert accuracy(y_true, y_pred) == 0.75


class TestConfusionMatrix:

    def test_confusion_matrix(self):
        y_true = np.array([[1, 1, 0, 0]])
        y_pred = np.array([[1, 0, 1, 0]])

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred)

        assert tn == 1
        assert fp == 1
        assert fn == 1
        assert tp == 1


class TestPrecision:

    def test_precision(self):
        y_true = np.array([[1, 1, 0, 0]])
        y_pred = np.array([[1, 0, 1, 0]])

        assert precision(y_true, y_pred) == 0.5

    def test_zero_precision(self):
        y_true = np.array([[1, 1, 1]])
        y_pred = np.array([[0, 0, 0]])

        assert precision(y_true, y_pred) == 0.0


class TestRecall:

    def test_recall(self):
        y_true = np.array([[1, 1, 0, 0]])
        y_pred = np.array([[1, 0, 1, 0]])

        assert recall(y_true, y_pred) == 0.5

    def test_zero_recall(self):
        y_true = np.array([[1, 1, 1]])
        y_pred = np.array([[0, 0, 0]])

        assert recall(y_true, y_pred) == 0.0


class TestF1Score:

    def test_f1_score(self):
        y_true = np.array([[1, 1, 0, 0]])
        y_pred = np.array([[1, 0, 1, 0]])

        assert f1_score(y_true, y_pred) == 0.5

    def test_zero_f1(self):
        y_true = np.array([[1, 1, 1]])
        y_pred = np.array([[0, 0, 0]])

        assert f1_score(y_true, y_pred) == 0.0
