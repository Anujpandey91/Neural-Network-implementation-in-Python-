import numpy as np


class Dropout:

    def __init__(self, keep_probability: float = 0.8):
        if not 0 < keep_probability <= 1:
            raise ValueError("keep_probability must be between 0 and 1.")

        self.keep_probability = keep_probability

    def forward(self, A, training=True):

        if not training:
            return A, None

        mask = np.random.rand(*A.shape) < self.keep_probability
        A_dropout = (A * mask) / self.keep_probability

        return A_dropout, mask

    def backward(self, dA, mask):

        if mask is None:
            return dA

        return (dA * mask) / self.keep_probability
