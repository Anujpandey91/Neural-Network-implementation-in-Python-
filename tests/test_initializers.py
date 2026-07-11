import numpy as np

from mydl.initializers import (
    zeros,
    ones,
    random,
)


class TestZeros:

    def test_shape(self):
        Z = zeros((3, 4))

        assert Z.shape == (3, 4)

    def test_all_zeros(self):
        Z = zeros((5, 2))

        assert np.all(Z == 0)


class TestOnes:

    def test_shape(self):
        O = ones((4, 6))

        assert O.shape == (4, 6)

    def test_all_ones(self):
        O = ones((2, 5))

        assert np.all(O == 1)


class TestRandom:

    def test_shape(self):
        np.random.seed(42)

        R = random((3, 4), 0.01)

        assert R.shape == (3, 4)

    def test_not_all_zero(self):
        np.random.seed(42)

        R = random((10, 10), 0.01)

        assert not np.all(R == 0)

    def test_scale(self):
        np.random.seed(42)

        scale = 0.01

        R = random((1000, 1000), scale)

        std = np.std(R)

        assert np.isclose(std, scale, atol=1e-3)

    def test_randomness(self):
        np.random.seed(42)
        R1 = random((5, 5), 0.01)

        np.random.seed(43)
        R2 = random((5, 5), 0.01)

        assert not np.array_equal(R1, R2)
