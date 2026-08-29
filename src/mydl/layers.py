import numpy as np


def linear_forward(
    A_prev: np.ndarray, W: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, tuple]:
    """Compute the linear forward propagation for a single layer."""
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)

    return Z, cache


def linear_backward(dZ: np.ndarray, cache: tuple):
    """Compute the linear portion of backward propagation for a layer.

    The incoming dZ is the gradient of the mean loss with respect to Z,
    so this function does not apply an additional 1/m scaling factor.
    """
    A_prev, W, b = cache

    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    A = activation(Z)
    activation_cache = Z
    cache = (linear_cache, activation_cache)

    return A, cache


def linear_activation_backward(dA, cache, activation):
    """Compute backward propagation for a linear->activation layer."""
    linear_cache, activation_cache = cache

    dZ = activation(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db
