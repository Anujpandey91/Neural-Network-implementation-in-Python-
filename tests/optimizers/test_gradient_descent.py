import numpy as np

from mydl.optimizers.gradient_descent import GradientDescent


class TestGradientDescent:

    def test_initialization(self):
        optimizer = GradientDescent(learning_rate=0.1)

        assert optimizer.learning_rate == 0.1

    def test_update(self):
        parameters = {
            "W1": np.array([[2.0]]),
            "b1": np.array([[1.0]]),
        }

        grads = {
            "dW1": np.array([[0.5]]),
            "db1": np.array([[0.2]]),
        }

        optimizer = GradientDescent(learning_rate=0.1)

        optimizer.update(parameters, grads)

        expected_W = np.array([[1.95]])
        expected_b = np.array([[0.98]])

        np.testing.assert_allclose(parameters["W1"], expected_W)

        np.testing.assert_allclose(parameters["b1"], expected_b)

    def test_multiple_layers(self):
        parameters = {
            "W1": np.array([[2.0]]),
            "b1": np.array([[1.0]]),
            "W2": np.array([[3.0]]),
            "b2": np.array([[2.0]]),
        }

        grads = {
            "dW1": np.array([[0.5]]),
            "db1": np.array([[0.2]]),
            "dW2": np.array([[0.4]]),
            "db2": np.array([[0.3]]),
        }

        optimizer = GradientDescent(learning_rate=0.1)

        optimizer.update(parameters, grads)

        np.testing.assert_allclose(parameters["W1"], np.array([[1.95]]))

        np.testing.assert_allclose(parameters["b1"], np.array([[0.98]]))

        np.testing.assert_allclose(parameters["W2"], np.array([[2.96]]))

        np.testing.assert_allclose(parameters["b2"], np.array([[1.97]]))
