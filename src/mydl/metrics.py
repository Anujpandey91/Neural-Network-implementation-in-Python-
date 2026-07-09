import numpy as np


def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute the classification accuracy.
    """
    
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Compute the confusion matrix for binary classification.

    Returns
    -------
    tuple
        (true_negative, false_positive,
         false_negative, true_positive)
    """

    true_negative = np.sum((y_true == 0) & (y_pred == 0))

    false_positive = np.sum((y_true == 0) & (y_pred == 1))

    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    true_positive = np.sum((y_true == 1) & (y_pred == 1))

    return (
        int(true_negative),
        int(false_positive),
        int(false_negative),
        int(true_positive),
    )


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute precision.
    """

    _, fp, _, tp = confusion_matrix(y_true, y_pred)

    denominator = tp + fp

    if denominator == 0:
        return 0.0

    return float(tp / denominator)


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute recall.
    """

    _, _, fn, tp = confusion_matrix(y_true, y_pred)

    denominator = tp + fn

    if denominator == 0:
        return 0.0

    return float(tp / denominator)


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute the F1 score.
    """

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    denominator = p + r

    if denominator == 0:
        return 0.0

    return float((2 * p * r) / denominator)


