import numpy as np

from mydl.regularizers import L1, L2, L1L2


class TestL2:

    def test_penalty(self):
        parameters = {
            "W1": np.array(
                [
                    [1.0, -2.0],
                    [3.0, -4.0],
                ]
            ),
            "b1": np.zeros((2, 1)),
            "W2": np.array(
                [
                    [5.0, -6.0],
                ]
            ),
            "b2": np.zeros((1, 1)),
        }

        regularizer = L2(lambda_=2.0)

        penalty = regularizer.penalty(parameters, m=10)

        np.testing.assert_allclose(penalty, 9.1)

    def test_gradient(self):
        W = np.array(
            [
                [1.0, -2.0],
                [3.0, -4.0],
            ]
        )

        regularizer = L2(lambda_=2.0)

        gradient = regularizer.gradient(W, m=10)

        expected = np.array(
            [
                [0.2, -0.4],
                [0.6, -0.8],
            ]
        )

        np.testing.assert_allclose(gradient, expected)


class TestL1:

    def test_penalty(self):
        parameters = {
            "W1": np.array(
                [
                    [1.0, -2.0],
                    [3.0, -4.0],
                ]
            ),
            "b1": np.zeros((2, 1)),
            "W2": np.array(
                [
                    [5.0, -6.0],
                ]
            ),
            "b2": np.zeros((1, 1)),
        }

        regularizer = L1(lambda_=2.0)

        penalty = regularizer.penalty(parameters, m=10)

        assert penalty == 4.2

    def test_gradient(self):
        W = np.array(
            [
                [1.0, -2.0],
                [3.0, -4.0],
            ]
        )

        regularizer = L1(lambda_=2.0)

        gradient = regularizer.gradient(W, m=10)

        expected = np.array(
            [
                [0.2, -0.2],
                [0.2, -0.2],
            ]
        )

        np.testing.assert_allclose(gradient, expected)


class TestL1L2:

    def test_penalty(self):
        parameters = {
            "W1": np.array(
                [
                    [1.0, -2.0],
                    [3.0, -4.0],
                ]
            ),
            "b1": np.zeros((2, 1)),
            "W2": np.array(
                [
                    [5.0, -6.0],
                ]
            ),
            "b2": np.zeros((1, 1)),
        }

        regularizer = L1L2(
            l1_lambda=2.0,
            l2_lambda=2.0,
        )

        penalty = regularizer.penalty(parameters, m=10)

        assert penalty == 13.3

    def test_gradient(self):
        W = np.array(
            [
                [1.0, -2.0],
                [3.0, -4.0],
            ]
        )

        regularizer = L1L2(
            l1_lambda=2.0,
            l2_lambda=2.0,
        )

        gradient = regularizer.gradient(W, m=10)

        expected = np.array(
            [
                [0.4, -0.6],
                [0.8, -1.0],
            ]
        )

        np.testing.assert_allclose(gradient, expected)
