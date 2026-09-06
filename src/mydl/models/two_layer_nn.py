import numpy as np
from ..initializers import zeros, random
from ..layers import linear_activation_forward, linear_activation_backward
from ..activations import relu, sigmoid, sigmoid_backward, relu_backward
from ..losses import binary_cross_entropy, binary_cross_entropy_backward
from ..metrics import accuracy


class TwoLayerNN:

    def __init__(
        self,
        hidden_units: int = 16,
        learning_rate: float = 0.01,
        epochs: int = 100,
        threshold: float = 0.5,
        verbose=True,
    ):
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self.cost_history = []
        self.is_fitted = False
        self.verbose = verbose

    def _initialize_parameters(self, n_features):
        self.W1 = random(
            (self.hidden_units, n_features),
            np.sqrt(2 / n_features),
        )
        self.b1 = zeros((self.hidden_units, 1))
        self.W2 = random(
            (1, self.hidden_units),
            np.sqrt(2 / self.hidden_units),
        )
        self.b2 = zeros((1, 1))

    def _forward(self, X: np.ndarray):
        A1, cache1 = linear_activation_forward(X, self.W1, self.b1, relu)
        A2, cache2 = linear_activation_forward(A1, self.W2, self.b2, sigmoid)

        caches = (cache1, cache2)
        self.A2 = A2

        return A2, caches

    def _compute_loss(self, Y, A2):
        return binary_cross_entropy(Y, A2)

    def _backward(self, Y, A2, caches):
        dA2 = binary_cross_entropy_backward(Y, A2)
        dA1, dW2, db2 = linear_activation_backward(
            dA2,
            caches[1],
            sigmoid_backward,
        )
        _, dW1, db1 = linear_activation_backward(
            dA1,
            caches[0],
            relu_backward,
        )

        self.dW1 = dW1
        self.db1 = db1
        self.dW2 = dW2
        self.db2 = db2

    def _update_parameters(self):
        self.W1 = self.W1 - self.learning_rate * self.dW1
        self.b1 = self.b1 - self.learning_rate * self.db1
        self.W2 = self.W2 - self.learning_rate * self.dW2
        self.b2 = self.b2 - self.learning_rate * self.db2

    def fit(self, X: np.ndarray, Y: np.ndarray):
        n_features, m = X.shape

        self._initialize_parameters(n_features)

        for epoch in range(self.epochs):
            # Forward propagation
            prediction, caches = self._forward(X)

            # Compute loss
            cost = self._compute_loss(Y, prediction)

            # Backward propagation
            self._backward(Y, prediction, caches)

            # Gradient descent
            self._update_parameters()

            # Save cost
            self.cost_history.append(cost)

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Cost: {cost:.6f}")

        self.is_fitted = True

        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        A2, _ = self._forward(X)

        return A2

    def predict(self, X):
        probability = self.predict_proba(X)
        predictions = (probability >= self.threshold).astype(int)

        return predictions

    def score(self, X, Y):
        predictions = self.predict(X)
        score = accuracy(Y, predictions)

        return score
