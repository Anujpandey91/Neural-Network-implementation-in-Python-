import numpy as np

from mydl.activations import (
    sigmoid,
    relu,
    tanh,
    
)


def test_sigmoid():
    """Test the sigmoid activation."""

    x = np.array([[-1.0, 0.0, 1.0]])

    expected = np.array([[0.26894142, 0.50000000, 0.73105858]])

    output = sigmoid(x)

    assert np.allclose(output, expected)


def test_relu():
    """Test the ReLU activation."""

    x = np.array([[-2.0, 0.0, 3.0]])

    expected = np.array([[0.0, 0.0, 3.0]])

    output = relu(x)

    assert np.array_equal(output, expected)


def test_tanh():
    """Test the tanh activation."""

    x = np.array([[-1.0, 0.0, 1.0]])

    expected = np.tanh(x)

    output = tanh(x)

    assert np.allclose(output, expected)

