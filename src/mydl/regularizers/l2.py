import numpy as np 

class L2:

    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def penalty(self, parameters, m):

        coefficient = (self.lambda_) *  (1 / (2 * m))
        L = len(parameters) // 2

        penalty = 0
        for l in range(1,L+1):
            squared_sum = np.sum(np.square(parameters["W" + str(l)]))
            penalty += coefficient * squared_sum

        return penalty

    def gradient(self, W, m):
        return (self.lambda_ / m) * W