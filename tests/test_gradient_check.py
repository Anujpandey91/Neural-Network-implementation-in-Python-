import numpy as np

from mydl.activations import relu, sigmoid
from mydl.layers import linear_activation_forward, linear_activation_backward
from mydl.losses import binary_cross_entropy


def numerical_gradient(loss_fn, parameter, epsilon=1e-7):
    """Compute a numerical gradient using the centered finite difference."""
    gradient = np.zeros_like(parameter)

    for index in np.ndindex(parameter.shape):
        original_value = parameter[index]

        parameter[index] = original_value + epsilon
        plus_loss = loss_fn()

        parameter[index] = original_value - epsilon
        minus_loss = loss_fn()

        parameter[index] = original_value

        gradient[index] = (plus_loss - minus_loss) / (2 * epsilon)

    return gradient


def relative_error(analytical, numerical):
    numerator = np.linalg.norm(analytical - numerical)
    denominator = np.linalg.norm(analytical) + np.linalg.norm(numerical)

    return numerator / max(denominator, 1e-15)


def test_two_layer_backward_gradient_check():
    np.random.seed(42)

    X = np.random.randn(3, 5)
    Y = np.array([[1, 0, 1, 0, 1]], dtype=float)

    W1 = np.random.randn(4, 3) * 0.1
    b1 = np.random.randn(4, 1) * 0.1
    W2 = np.random.randn(1, 4) * 0.1
    b2 = np.random.randn(1, 1) * 0.1

    def forward():
        A1, cache1 = linear_activation_forward(X, W1, b1, relu)
        A2, cache2 = linear_activation_forward(A1, W2, b2, sigmoid)
        return A2, (cache1, cache2)

    A2, caches = forward()

    dA2 = -(1 / X.shape[1]) * (
        Y / A2 - (1 - Y) / (1 - A2)
    )

    dA1, dW2, db2 = linear_activation_backward(
        dA2,
        caches[1],
        lambda dA, Z: dA * sigmoid(Z) * (1 - sigmoid(Z)),
    )
    _, dW1, db1 = linear_activation_backward(
        dA1,
        caches[0],
        lambda dA, Z: dA * (Z > 0),
    )

    def loss():
        prediction, _ = forward()
        return binary_cross_entropy(Y, prediction)

    numerical_dW1 = numerical_gradient(loss, W1)
    numerical_db1 = numerical_gradient(loss, b1)
    numerical_dW2 = numerical_gradient(loss, W2)
    numerical_db2 = numerical_gradient(loss, b2)

    assert relative_error(dW1, numerical_dW1) < 1e-6
    assert relative_error(db1, numerical_db1) < 1e-6
    assert relative_error(dW2, numerical_dW2) < 1e-6
    assert relative_error(db2, numerical_db2) < 1e-6
