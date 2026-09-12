import numpy as np

from mydl.optimizers.rmsprop import RMSProp


class TestRMSProp:

    def test_initialization(self):
        optimizer = RMSProp(learning_rate=0.1, beta=0.9, epsilon=1e-8)

        assert optimizer.learning_rate == 0.1
        assert optimizer.beta == 0.9
        assert optimizer.epsilon == 1e-8
        assert optimizer.squared_gradients == {}

    def test_first_update(self):
        parameters = {
            "W1": np.array([[2.0]]),
            "b1": np.array([[1.0]]),
        }

        grads = {
            "dW1": np.array([[0.5]]),
            "db1": np.array([[0.2]]),
        }

        optimizer = RMSProp(learning_rate=0.1, beta=0.9, epsilon=1e-8)

        optimizer.update(parameters, grads)

        # S_dW = 0.9(0) + 0.1(0.5^2) = 0.025
        expected_s_dW = np.array([[0.025]])

        # S_db = 0.9(0) + 0.1(0.2^2) = 0.004
        expected_s_db = np.array([[0.004]])

        expected_W = np.array([[1.6837722972286964]])
        expected_b = np.array([[0.6837726292671285]])

        np.testing.assert_allclose(optimizer.squared_gradients["S_dW1"], expected_s_dW)

        np.testing.assert_allclose(optimizer.squared_gradients["S_db1"], expected_s_db)

        np.testing.assert_allclose(parameters["W1"], expected_W)

        np.testing.assert_allclose(parameters["b1"], expected_b)

    def test_state_persists_between_updates(self):
        parameters = {
            "W1": np.array([[2.0]]),
            "b1": np.array([[1.0]]),
        }

        grads = {
            "dW1": np.array([[0.5]]),
            "db1": np.array([[0.2]]),
        }

        optimizer = RMSProp(learning_rate=0.1, beta=0.9, epsilon=1e-8)

        optimizer.update(parameters, grads)
        optimizer.update(parameters, grads)

        # S2 = 0.9(0.025) + 0.1(0.25)
        #    = 0.0475
        expected_s_dW = np.array([[0.0475]])

        # S2 = 0.9(0.004) + 0.1(0.04)
        #    = 0.0076
        expected_s_db = np.array([[0.0076]])

        expected_W = np.array([[1.4543565875071554]])
        expected_b = np.array([[0.4543570463278216]])

        np.testing.assert_allclose(optimizer.squared_gradients["S_dW1"], expected_s_dW)

        np.testing.assert_allclose(optimizer.squared_gradients["S_db1"], expected_s_db)

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

        optimizer = RMSProp(learning_rate=0.1, beta=0.9, epsilon=1e-8)

        optimizer.update(parameters, grads)

        np.testing.assert_allclose(
            optimizer.squared_gradients["S_dW1"], np.array([[0.025]])
        )

        np.testing.assert_allclose(
            optimizer.squared_gradients["S_db1"], np.array([[0.004]])
        )

        np.testing.assert_allclose(
            optimizer.squared_gradients["S_dW2"], np.array([[0.016]])
        )

        np.testing.assert_allclose(
            optimizer.squared_gradients["S_db2"], np.array([[0.009]])
        )
