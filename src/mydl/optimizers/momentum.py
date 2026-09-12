import numpy as np

class Momentum:
    
    def __init__(self, learning_rate: float=0.01, beta: float=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        
        self.velocity = {}
        
    def update(self, parameters, grads):
        
        L = len(parameters) // 2
        
        if not self.velocity:
            for l in range(1, L+1):
                self.velocity["V_dW" + str(l)] = np.zeros_like(grads["dW" + str(l)])
                self.velocity["V_db" + str(l)] = np.zeros_like(grads["db" + str(l)])
                
        for l in range(1, L+1):
            self.velocity["V_dW" + str(l)] = self.beta * self.velocity["V_dW" + str(l)] + (1-self.beta) * grads["dW" + str(l)]
            self.velocity["V_db" + str(l)] = self.beta * self.velocity["V_db" + str(l)] + (1-self.beta) * grads["db" + str(l)]
            
            parameters["W" + str(l)] = parameters["W" + str(l)] - self.learning_rate * self.velocity["V_dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - self.learning_rate * self.velocity["V_db" + str(l)]