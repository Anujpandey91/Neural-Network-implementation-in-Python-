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

    loss = binary_cross_entropy(y_true, y_pred)

    expected = -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

    assert isinstance(loss, float)
    assert np.isclose(loss, expected)


def test_binary_cross_entropy_backward():
    y_true = np.array([[1, 0, 1]])
    y_pred = np.array([[0.9, 0.2, 0.8]])

    grad = binary_cross_entropy_backward(y_true, y_pred)

    expected = -(y_true / y_pred - (1 - y_true) / (1 - y_pred))

    assert grad.shape == y_pred.shape
    assert np.allclose(grad, expected)


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

    loss = categorical_cross_entropy(y_true, y_pred)

    expected = -(1 / 2) * np.sum(y_true * np.log(y_pred))

    assert isinstance(loss, float)
    assert np.isclose(loss, expected)


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

    grad = categorical_cross_entropy_backward(y_true, y_pred)
    expected = -(y_true / y_pred) / y_true.shape[1]

    assert grad.shape == y_pred.shape
    assert np.allclose(grad, expected)


def test_mean_squared_error():
    y_true = np.array([[1, 2, 3]])
    y_pred = np.array([[1.1, 1.9, 3.2]])

    loss = mean_squared_error(y_true, y_pred)

    expected = np.mean((y_pred - y_true) ** 2)

    assert isinstance(loss, float)
    assert np.isclose(loss, expected)


def test_mean_squared_error_backward():
    y_true = np.array([[1, 2, 3]])
    y_pred = np.array([[1.1, 1.9, 3.2]])

    grad = mean_squared_error_backward(y_true, y_pred)
    expected = (2 / y_true.shape[1]) * (y_pred - y_true)

    assert grad.shape == y_pred.shape
    assert np.allclose(grad, expected)


def test_mean_absolute_error():
    y_true = np.array([[1, 2, 3]])
    y_pred = np.array([[1.1, 1.9, 3.2]])

    loss = mean_absolute_error(y_true, y_pred)

    expected = np.mean(np.abs(y_pred - y_true))

    assert isinstance(loss, float)
    assert np.isclose(loss, expected)


def test_mean_absolute_error_backward():
    y_true = np.array([[1, 2, 3]])
    y_pred = np.array([[1.1, 1.9, 3.2]])

    grad = mean_absolute_error_backward(y_true, y_pred)
    expected = (1 / y_true.shape[1]) * np.sign(y_pred - y_true)

    assert grad.shape == y_pred.shape
    assert np.allclose(grad, expected)
