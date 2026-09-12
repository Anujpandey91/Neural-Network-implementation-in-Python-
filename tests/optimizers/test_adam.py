import numpy as np

from mydl.optimizers.adam import Adam


class TestAdam:

    def test_initialization(self):
        optimizer = Adam(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

        assert optimizer.learning_rate == 0.01
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.999
        assert optimizer.epsilon == 1e-8

        assert optimizer.velocity == {}
        assert optimizer.squared_gradients == {}
        assert optimizer.t == 0

    def test_first_update(self):
        parameters = {
            "W1": np.array([[2.0]]),
            "b1": np.array([[1.0]]),
        }

        grads = {
            "dW1": np.array([[0.5]]),
            "db1": np.array([[0.2]]),
        }

        optimizer = Adam(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

        optimizer.update(parameters, grads)

        # t should now be 1
        assert optimizer.t == 1

        # First moment
        # v_dW = 0.9(0) + 0.1(0.5) = 0.05
        expected_v_dW = np.array([[0.05]])

        # v_db = 0.9(0) + 0.1(0.2) = 0.02
        expected_v_db = np.array([[0.02]])

        # Second moment
        # s_dW = 0.999(0) + 0.001(0.5^2) = 0.00025
        expected_s_dW = np.array([[0.00025]])

        # s_db = 0.999(0) + 0.001(0.2^2) = 0.00004
        expected_s_db = np.array([[0.00004]])

        np.testing.assert_allclose(optimizer.velocity["V_dW1"], expected_v_dW)

        np.testing.assert_allclose(optimizer.velocity["V_db1"], expected_v_db)

        np.testing.assert_allclose(optimizer.squared_gradients["S_dW1"], expected_s_dW)

        np.testing.assert_allclose(optimizer.squared_gradients["S_db1"], expected_s_db)

        # Bias-corrected values at t = 1
        expected_v_dW_corrected = np.array([[0.5]])
        expected_v_db_corrected = np.array([[0.2]])

        expected_s_dW_corrected = np.array([[0.25]])
        expected_s_db_corrected = np.array([[0.04]])

        v_dW_corrected = optimizer.velocity["V_dW1"] / (
            1 - optimizer.beta1**optimizer.t
        )

        v_db_corrected = optimizer.velocity["V_db1"] / (
            1 - optimizer.beta1**optimizer.t
        )

        s_dW_corrected = optimizer.squared_gradients["S_dW1"] / (
            1 - optimizer.beta2**optimizer.t
        )

        s_db_corrected = optimizer.squared_gradients["S_db1"] / (
            1 - optimizer.beta2**optimizer.t
        )

        np.testing.assert_allclose(v_dW_corrected, expected_v_dW_corrected)

        np.testing.assert_allclose(v_db_corrected, expected_v_db_corrected)

        np.testing.assert_allclose(s_dW_corrected, expected_s_dW_corrected)

        np.testing.assert_allclose(s_db_corrected, expected_s_db_corrected)

        # Final parameter values
        expected_W = np.array([[1.9900000002]])
        expected_b = np.array([[0.99000000125]])

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

        optimizer = Adam(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

        optimizer.update(parameters, grads)
        optimizer.update(parameters, grads)

        assert optimizer.t == 2

        # v2
        expected_v_dW = np.array([[0.095]])
        expected_v_db = np.array([[0.038]])

        # s2
        expected_s_dW = np.array([[0.00049975]])
        expected_s_db = np.array([[0.00007996]])

        np.testing.assert_allclose(optimizer.velocity["V_dW1"], expected_v_dW)

        np.testing.assert_allclose(optimizer.velocity["V_db1"], expected_v_db)

        np.testing.assert_allclose(optimizer.squared_gradients["S_dW1"], expected_s_dW)

        np.testing.assert_allclose(optimizer.squared_gradients["S_db1"], expected_s_db)

        expected_W = np.array([[1.9800000004]])
        expected_b = np.array([[0.9800000025]])

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

        optimizer = Adam(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

        optimizer.update(parameters, grads)

        assert "V_dW1" in optimizer.velocity
        assert "V_db1" in optimizer.velocity
        assert "V_dW2" in optimizer.velocity
        assert "V_db2" in optimizer.velocity

        assert "S_dW1" in optimizer.squared_gradients
        assert "S_db1" in optimizer.squared_gradients
        assert "S_dW2" in optimizer.squared_gradients
        assert "S_db2" in optimizer.squared_gradients

        np.testing.assert_allclose(optimizer.velocity["V_dW1"], np.array([[0.05]]))

        np.testing.assert_allclose(optimizer.velocity["V_db1"], np.array([[0.02]]))

        np.testing.assert_allclose(optimizer.velocity["V_dW2"], np.array([[0.04]]))

        np.testing.assert_allclose(optimizer.velocity["V_db2"], np.array([[0.03]]))

        np.testing.assert_allclose(
            optimizer.squared_gradients["S_dW1"], np.array([[0.00025]])
        )

        np.testing.assert_allclose(
            optimizer.squared_gradients["S_db1"], np.array([[0.00004]])
        )

        np.testing.assert_allclose(
            optimizer.squared_gradients["S_dW2"], np.array([[0.00016]])
        )

        np.testing.assert_allclose(
            optimizer.squared_gradients["S_db2"], np.array([[0.00009]])
        )
