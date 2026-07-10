import numpy as np


def linear_forward(
    A_prev: np.ndarray, W: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, tuple]:
    """
    Compute the linear forward propagation for a single layer.

    Parameters
    ----------
    A_prev : np.ndarray
        Activations from the previous layer.

    W : np.ndarray
        Weight matrix.

    b : np.ndarray
        Bias vector.

    Returns
    -------
    Z : np.ndarray
        Linear output before applying the activation function.

    cache : tuple
        Cached values (A_prev, W, b) required for backward propagation.
    """

    Z = np.dot(W, A_prev) + b

    cache = (A_prev, W, b)

    return Z, cache


def linear_backward(dZ: np.ndarray, cache: tuple):
    """
    Compute the linear portion of backward propagation for a single layer.

    Parameters
    ----------
    dZ : np.ndarray
        Gradient of the loss with respect to the linear output Z.

    cache : tuple
        Cached values (A_prev, W, b) from the forward pass.

    Returns
    -------
    dA_prev : np.ndarray
        Gradient of the loss with respect to the previous layer activations.

    dW : np.ndarray
        Gradient of the loss with respect to the weights.

    db : np.ndarray
        Gradient of the loss with respect to the bias.
    """

    A_prev, W, b = cache

    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)

    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_forward(A_prev, W, b, activation):

    Z, linear_cache = linear_forward(A_prev, W, b)

    A = activation(Z)

    activation_cache = Z

    cache = (linear_cache, activation_cache)

    return A, cache


def linear_activation_backward(dA, cache, activation):
    """
    Compute the backward propagation for the linear->activation layer.

    Parameters
    ----------
    dA : np.ndarray
        Gradient of the loss with respect to the activation output.

    cache : tuple
        Tuple containing (linear_cache, activation_cache).

    activation : callable
        Activation backward function (e.g. sigmoid_backward, relu_backward).

    Returns
    -------
    dA_prev : np.ndarray
        Gradient with respect to the previous layer activation.

    dW : np.ndarray
        Gradient with respect to the weights.

    db : np.ndarray
        Gradient with respect to the bias.
    """

    linear_cache, activation_cache = cache

    dZ = activation(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db
