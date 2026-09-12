class GradientDescent:

    def __init__(self, learning_rate: float=0.01):
        self.learning_rate = learning_rate

    def update(self, parameters, grads):

        L = len(parameters) // 2

        for l in range(1, L + 1):
            parameters["W" + str(l)] = (
                parameters["W" + str(l)] - self.learning_rate * grads["dW" + str(l)]
            )
            parameters["b" + str(l)] = (
                parameters["b" + str(l)] - self.learning_rate * grads["db" + str(l)]
            )
