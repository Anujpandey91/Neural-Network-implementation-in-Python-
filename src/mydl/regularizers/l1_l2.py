import numpy as np


class L1L2:

    def __init__(self, l1_lambda=0.0, l2_lambda=0.0):
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def penalty(self, parameters, m):

        L = len(parameters) // 2

        l1_penalty = 0
        l2_penalty = 0

        for l in range(1, L + 1):
            W = parameters["W" + str(l)]

            l1_penalty += np.sum(np.abs(W))
            l2_penalty += np.sum(np.square(W))

        l1_penalty *= self.l1_lambda / m
        l2_penalty *= self.l2_lambda / (2 * m)

        return l1_penalty + l2_penalty

    def gradient(self, W, m):

        l1_gradient = (self.l1_lambda / m) * np.sign(W)
        l2_gradient = (self.l2_lambda / m) * W

        return l1_gradient + l2_gradient
