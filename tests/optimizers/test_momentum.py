import numpy as np

from mydl.optimizers.momentum import Momentum


class TestMomentum:

    def test_initialization(self):
        optimizer = Momentum(learning_rate=0.1, beta=0.9)

        assert optimizer.learning_rate == 0.1
        assert optimizer.beta == 0.9
        assert optimizer.velocity == {}

    def test_first_update(self):
        parameters = {
            "W1": np.array([[2.0]]),
            "b1": np.array([[1.0]]),
        }

        grads = {
            "dW1": np.array([[0.5]]),
            "db1": np.array([[0.2]]),
        }

        optimizer = Momentum(learning_rate=0.1, beta=0.9)

        optimizer.update(parameters, grads)

        expected_W = np.array([[1.995]])
        expected_b = np.array([[0.998]])

        expected_v_dW = np.array([[0.05]])
        expected_v_db = np.array([[0.02]])

        np.testing.assert_allclose(parameters["W1"], expected_W)

        np.testing.assert_allclose(parameters["b1"], expected_b)

        np.testing.assert_allclose(optimizer.velocity["V_dW1"], expected_v_dW)

        np.testing.assert_allclose(optimizer.velocity["V_db1"], expected_v_db)

    def test_state_persists_between_updates(self):
        parameters = {
            "W1": np.array([[2.0]]),
            "b1": np.array([[1.0]]),
        }

        grads = {
            "dW1": np.array([[0.5]]),
            "db1": np.array([[0.2]]),
        }

        optimizer = Momentum(learning_rate=0.1, beta=0.9)

        optimizer.update(parameters, grads)
        optimizer.update(parameters, grads)

        # Second velocity:
        # v2 = 0.9(0.05) + 0.1(0.5) = 0.095
        expected_v_dW = np.array([[0.095]])

        # Second bias velocity:
        # v2 = 0.9(0.02) + 0.1(0.2) = 0.038
        expected_v_db = np.array([[0.038]])

        # W2 = 1.995 - 0.1(0.095) = 1.9855
        expected_W = np.array([[1.9855]])

        # b2 = 0.998 - 0.1(0.038) = 0.9942
        expected_b = np.array([[0.9942]])

        np.testing.assert_allclose(optimizer.velocity["V_dW1"], expected_v_dW)

        np.testing.assert_allclose(optimizer.velocity["V_db1"], expected_v_db)

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

        optimizer = Momentum(learning_rate=0.1, beta=0.9)

        optimizer.update(parameters, grads)

        np.testing.assert_allclose(optimizer.velocity["V_dW1"], np.array([[0.05]]))

        np.testing.assert_allclose(optimizer.velocity["V_db1"], np.array([[0.02]]))

        np.testing.assert_allclose(optimizer.velocity["V_dW2"], np.array([[0.04]]))

        np.testing.assert_allclose(optimizer.velocity["V_db2"], np.array([[0.03]]))
