import numpy as np


class Adam:

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.velocity = {}
        self.squared_gradients = {}

        self.t = 0

    def update(self, parameters, grads):

        L = len(parameters) // 2

        # Initialize optimizer state
        if not self.velocity:
            for l in range(1, L + 1):
                self.velocity["V_dW" + str(l)] = np.zeros_like(grads["dW" + str(l)])
                self.velocity["V_db" + str(l)] = np.zeros_like(grads["db" + str(l)])
                self.squared_gradients["S_dW" + str(l)] = np.zeros_like(grads["dW" + str(l)])
                self.squared_gradients["S_db" + str(l)] = np.zeros_like(grads["db" + str(l)])
                                

        # One optimizer step
        self.t += 1

        for l in range(1, L + 1):

            # 1. Update v using Momentum equation
            self.velocity["V_dW" + str(l)] = self.beta1 * self.velocity["V_dW" + str(l)] + (1-self.beta1) * grads["dW" + str(l)]
            self.velocity["V_db" + str(l)] = self.beta1 * self.velocity["V_db" + str(l)] + (1-self.beta1) * grads["db" + str(l)]
            

            # 2. Update s using RMSProp equation
            self.squared_gradients["S_dW" + str(l)] = self.beta2 * self.squared_gradients["S_dW" + str(l)] + (1-self.beta2) * (grads["dW" + str(l)])**2
            self.squared_gradients["S_db" + str(l)] = self.beta2 * self.squared_gradients["S_db" + str(l)] + (1-self.beta2) * (grads["db" + str(l)])**2
             

            # 3. Bias correction
            V_dW_corrected = self.velocity["V_dW" + str(l)] / (1 - self.beta1 ** self.t)
            V_db_corrected = self.velocity["V_db" + str(l)] / (1 - self.beta1 ** self.t)
            S_dW_corrected = self.squared_gradients["S_dW" + str(l)] / (1 - self.beta2 ** self.t)
            S_db_corrected = self.squared_gradients["S_db" + str(l)] / (1 - self.beta2 ** self.t)

            # 4. Update W and b
            parameters["W" + str(l)] = parameters["W" + str(l)] - self.learning_rate * (V_dW_corrected / np.sqrt(S_dW_corrected + self.epsilon))
            parameters["b" + str(l)] = parameters["b" + str(l)] - self.learning_rate * (V_db_corrected / np.sqrt(S_db_corrected + self.epsilon)) 