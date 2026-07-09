import numpy as np

from mydl.losses import (
    binary_cross_entropy,
    binary_cross_entropy_backward,
    categorical_cross_entropy,
    categorical_cross_entropy_backward,
    mean_squared_error,
    mean_squared_error_backward,
    mean_absolute_error,
    mean_absolute_error_backward,
)


def test_binary_cross_entropy():

    y_true = np.array([[1, 0, 1]])

    y_pred = np.array([[0.9, 0.2, 0.8]])

    loss = binary_cross_entropy(y_pred, y_true)

    assert isinstance(loss, float)
    assert loss > 0


def test_binary_cross_entropy_backward():

    y_true = np.array([[1, 0, 1]])

    y_pred = np.array([[0.9, 0.2, 0.8]])

    grad = binary_cross_entropy_backward(y_pred, y_true)

    assert grad.shape == y_pred.shape


def test_categorical_cross_entropy():

    y_true = np.array(
        [
            [1, 0],
            [0, 1],
            [0, 0],
        ]
    )

    y_pred = np.array(
        [
            [0.8, 0.1],
            [0.1, 0.8],
            [0.1, 0.1],
        ]
    )

    loss = categorical_cross_entropy(y_pred, y_true)

    assert isinstance(loss, float)
    assert loss > 0


def test_categorical_cross_entropy_backward():

    y_true = np.array(
        [
            [1, 0],
            [0, 1],
            [0, 0],
        ]
    )

    y_pred = np.array(
        [
            [0.8, 0.1],
            [0.1, 0.8],
            [0.1, 0.1],
        ]
    )

    grad = categorical_cross_entropy_backward(y_pred, y_true)

    assert grad.shape == y_pred.shape


def test_mean_squared_error():

    y_true = np.array([[1, 2, 3]])

    y_pred = np.array([[1.1, 1.9, 3.2]])

    loss = mean_squared_error(y_pred, y_true)

    assert isinstance(loss, float)
    assert loss >= 0


def test_mean_squared_error_backward():

    y_true = np.array([[1, 2, 3]])

    y_pred = np.array([[1.1, 1.9, 3.2]])

    grad = mean_squared_error_backward(y_pred, y_true)

    assert grad.shape == y_pred.shape


def test_mean_absolute_error():

    y_true = np.array([[1, 2, 3]])

    y_pred = np.array([[1.1, 1.9, 3.2]])

    loss = mean_absolute_error(y_pred, y_true)

    assert isinstance(loss, float)
    assert loss >= 0


def test_mean_absolute_error_backward():

    y_true = np.array([[1, 2, 3]])

    y_pred = np.array([[1.1, 1.9, 3.2]])

    grad = mean_absolute_error_backward(y_pred, y_true)

    assert grad.shape == y_pred.shape
