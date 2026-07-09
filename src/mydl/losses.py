import numpy as np


def binary_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute the mean Binary Cross Entropy (BCE) loss.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels of shape (1, m).

    y_pred : np.ndarray
        Predicted probabilities of shape (1, m).

    Returns
    -------
    float
        Mean binary cross entropy loss.
    """

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    m = y_true.shape[1]

    cost = -(1 / m) * np.sum(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

    return float(cost)


def binary_cross_entropy_backward(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute the gradient of Binary Cross Entropy loss
    with respect to the predicted probabilities.
    """

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    dy_pred = -(np.divide(y_true, y_pred) - np.divide(1 - y_true, 1 - y_pred))

    return dy_pred


def categorical_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute the mean Categorical Cross Entropy loss.

    Parameters
    ----------
    y_true : np.ndarray
        One-hot encoded labels of shape (C, m).

    y_pred : np.ndarray
        Predicted probabilities of shape (C, m).

    Returns
    -------
    float
        Mean categorical cross entropy loss.
    """

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    m = y_true.shape[1]

    cost = -(1 / m) * np.sum(y_true * np.log(y_pred))

    return float(cost)


def categorical_cross_entropy_backward(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute the gradient of Categorical Cross Entropy
    with respect to the predicted probabilities.
    """

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    m = y_true.shape[1]

    dy_pred = -(y_true / y_pred) / m

    return dy_pred


def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute the Mean Squared Error (MSE).
    """

    m = y_true.shape[1]

    cost = (1 / m) * np.sum((y_pred - y_true) ** 2)

    return float(cost)


def mean_squared_error_backward(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute the gradient of Mean Squared Error
    with respect to the predictions.
    """

    m = y_true.shape[1]

    dy_pred = (2 / m) * (y_pred - y_true)

    return dy_pred


def mean_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute the Mean Absolute Error (MAE).
    """

    m = y_true.shape[1]

    cost = (1 / m) * np.sum(np.abs(y_pred - y_true))

    return float(cost)


def mean_absolute_error_backward(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute the gradient of Mean Absolute Error
    with respect to the predictions.
    """

    m = y_true.shape[1]

    dy_pred = (1 / m) * np.sign(y_pred - y_true)

    return dy_pred
