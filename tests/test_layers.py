import numpy as np

from mydl.layers import (
    linear_forward,
    linear_backward,
    linear_activation_forward,
    linear_activation_backward,
)

from mydl.activations import (
    relu,
    sigmoid,
    relu_backward,
    sigmoid_backward,
)


class TestLinearForward:

    def test_output_shape(self):
        np.random.seed(42)

        A_prev = np.random.randn(5, 10)
        W = np.random.randn(3, 5)
        b = np.random.randn(3, 1)

        Z, cache = linear_forward(A_prev, W, b)

        assert Z.shape == (3, 10)
        assert len(cache) == 3

    def test_output_values(self):
        A_prev = np.array([[1.0, 2.0], [3.0, 4.0]])

        W = np.array([[5.0, 6.0]])

        b = np.array([[1.0]])

        Z, _ = linear_forward(A_prev, W, b)

        expected = np.array([[24.0, 35.0]])

        np.testing.assert_allclose(Z, expected)


class TestLinearBackward:

    def test_shapes(self):
        np.random.seed(42)

        A_prev = np.random.randn(5, 10)
        W = np.random.randn(3, 5)
        b = np.random.randn(3, 1)

        cache = (A_prev, W, b)

        dZ = np.random.randn(3, 10)

        dA_prev, dW, db = linear_backward(dZ, cache)

        assert dA_prev.shape == A_prev.shape
        assert dW.shape == W.shape
        assert db.shape == b.shape


class TestLinearActivationForward:

    def test_relu_shape(self):
        np.random.seed(42)

        A_prev = np.random.randn(5, 8)
        W = np.random.randn(4, 5)
        b = np.random.randn(4, 1)

        A, cache = linear_activation_forward(
            A_prev,
            W,
            b,
            relu,
        )

        assert A.shape == (4, 8)
        assert len(cache) == 2

    def test_sigmoid_shape(self):
        np.random.seed(42)

        A_prev = np.random.randn(5, 8)
        W = np.random.randn(4, 5)
        b = np.random.randn(4, 1)

        A, cache = linear_activation_forward(
            A_prev,
            W,
            b,
            sigmoid,
        )

        assert A.shape == (4, 8)
        assert len(cache) == 2

    def test_sigmoid_output_range(self):
        np.random.seed(42)

        A_prev = np.random.randn(5, 8)
        W = np.random.randn(4, 5)
        b = np.random.randn(4, 1)

        A, _ = linear_activation_forward(
            A_prev,
            W,
            b,
            sigmoid,
        )

        assert np.all(A >= 0)
        assert np.all(A <= 1)


class TestLinearActivationBackward:

    def test_relu_shapes(self):
        np.random.seed(42)

        A_prev = np.random.randn(5, 10)
        W = np.random.randn(3, 5)
        b = np.random.randn(3, 1)

        A, cache = linear_activation_forward(
            A_prev,
            W,
            b,
            relu,
        )

        dA = np.random.randn(*A.shape)

        dA_prev, dW, db = linear_activation_backward(
            dA,
            cache,
            relu_backward,
        )

        assert dA_prev.shape == A_prev.shape
        assert dW.shape == W.shape
        assert db.shape == b.shape

    def test_sigmoid_shapes(self):
        np.random.seed(42)

        A_prev = np.random.randn(5, 10)
        W = np.random.randn(3, 5)
        b = np.random.randn(3, 1)

        A, cache = linear_activation_forward(
            A_prev,
            W,
            b,
            sigmoid,
        )

        dA = np.random.randn(*A.shape)

        dA_prev, dW, db = linear_activation_backward(
            dA,
            cache,
            sigmoid_backward,
        )

        assert dA_prev.shape == A_prev.shape
        assert dW.shape == W.shape
        assert db.shape == b.shape
