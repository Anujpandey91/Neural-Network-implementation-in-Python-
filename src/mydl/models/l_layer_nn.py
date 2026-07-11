import numpy as np 
from ..metrics import accuracy
from ..initializers import random, zeros
from ..layers import linear_activation_backward, linear_activation_forward
from ..activations import relu, sigmoid,sigmoid_backward, relu_backward
from ..losses import binary_cross_entropy, binary_cross_entropy_backward

class LLayerNN:

    def __init__(
        self,
        hidden_layer: list[int],
        learning_rate: int=0.01,
        epochs: int=100,
        threshold: float=0.5,
        verbose = True
    ):
        self.hidden_layer = hidden_layer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold

        self.cost_history = []
        self.is_fitted = False
        self.verbose = verbose

        self.parameters = {}
        self.grads = {}

    def _initialize_parameters(self):

        for l in range(1, len(self.layer_dims)):
            self.parameters["W" + str(l)] = random(
                (self.layer_dims[l], self.layer_dims[l - 1]),
                np.sqrt(2 / self.layer_dims[l - 1]),
            )
            self.parameters["b"+ str(l)] = zeros((self.layer_dims[l],1))

    def _forward(self, X):
        A = X
        L = int(len(self.parameters)/2)
        caches = []

        for l in range(1, L):
            W_l = self.parameters["W" + str(l)]
            b_l = self.parameters["b" + str(l)]
            A_i , cache = linear_activation_forward(A, W_l, b_l, relu)

            A = A_i
            caches.append(cache)

        W = self.parameters["W"+ str(L)]
        b = self.parameters["b"+ str(L)]
        AL, cache = linear_activation_forward(A, W, b, sigmoid)
        caches.append(cache)

        return AL, caches

        ...

    def _compute_loss(self, Y, A):
        return binary_cross_entropy(Y,A)

    def _backward(self, Y, A, caches):

        L = int(len(self.parameters)/2)

        dAL = binary_cross_entropy_backward(Y,A)

        dA_prev, dWL, dbL = linear_activation_backward(
            dAL,
            caches[L-1],
            sigmoid_backward
        )

        self.grads["dW" + str(L)] = dWL
        self.grads["db" + str(L)] = dbL

        for l in range(L-1, 0, -1):

            dA_prev, dWl, dbl = linear_activation_backward(
                dA_prev,
                caches[l-1],
                relu_backward
            )
            self.grads["dW" + str(l)] = dWl
            self.grads["db" + str(l)] = dbl

    def _update_parameters(self):

        L = len(self.parameters) // 2

        for l in range(1,L+1):
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - self.learning_rate * self.grads["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - self.learning_rate * self.grads["db" + str(l)]

    def fit(self, X:np.ndarray, Y: np.ndarray):
        n_features,m = X.shape

        self.layer_dims = [
            n_features,
            *self.hidden_layer,
            1
        ]
        self._initialize_parameters()

        for epoch in range(self.epochs):
            # forward propagation
            prediction , caches = self._forward(X)
            # compute loss
            cost = self._compute_loss(Y,prediction)
            # backward propagation
            self._backward(Y, prediction, caches)
            # gradient descent
            self._update_parameters()
            # save cost
            self.cost_history.append(cost)

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Cost: {cost:.6f}")    

        self.is_fitted = True

        return self

    def predict_proba(self,X):

        if not self.is_fitted:
            raise ValueError("Model has not been fitted.")

        A,_ = self._forward(X)

        return A

    def predict(self,X):

        if not self.is_fitted:
            raise ValueError("Model has not been fitted")

        probability = self.predict_proba(X)
        predictions = (probability >= self.threshold).astype(int)

        return predictions

    def score(self,X,Y):
        predictions = self.predict(X)
        score = accuracy(Y, predictions)

        return score 
