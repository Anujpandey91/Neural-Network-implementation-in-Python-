import numpy as np


class L1:

    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def penalty(self, parameters, m):

        coefficient = self.lambda_ / m
        L = len(parameters) // 2

        penalty = 0

        for l in range(1, L + 1):
            absolute_sum = np.sum(np.abs(parameters["W" + str(l)]))
            penalty += coefficient * absolute_sum

        return penalty

    def gradient(self, W, m):
        return (self.lambda_ / m) * np.sign(W)
