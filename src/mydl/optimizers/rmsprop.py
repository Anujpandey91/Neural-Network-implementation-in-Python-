import numpy as np

class RMSProp:
    
    def __init__(
        self,
        learning_rate: float=0.01,
        beta: float=0.9,
        epsilon: float=1e-8
    ):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.squared_gradients = {}
        
    def update(self, parameters, grads):
        
        L = len(parameters) // 2
        
        if not self.squared_gradients:
            for l in range(1, L+1):
                self.squared_gradients["S_dW" + str(l)] = np.zeros_like(grads["dW" + str(l)])
                self.squared_gradients["S_db" + str(l)] = np.zeros_like(grads["db" + str(l)])
                
        for l in range(1, L+1):
            self.squared_gradients["S_dW" + str(l)] = self.beta * self.squared_gradients["S_dW" + str(l)] + (1-self.beta) * (grads["dW" + str(l)])**2
            self.squared_gradients["S_db" + str(l)] = self.beta * self.squared_gradients["S_db" + str(l)] + (1-self.beta) * (grads["db" + str(l)])**2
            
            S_dW = self.squared_gradients["S_dW" + str(l)]
            S_db = self.squared_gradients["S_db" + str(l)]
            epsilon = self.epsilon
            dW = grads["dW" + str(l)]
            db = grads["db" + str(l)]
            
            parameters["W" + str(l)] = parameters["W" + str(l)] - self.learning_rate * (dW / np.sqrt(S_dW + epsilon)) 
            parameters["b" + str(l)] = parameters["b" + str(l)] - self.learning_rate * (db / np.sqrt(S_db + epsilon))