import numpy as np
from ..initializers import zeros
from ..activations import sigmoid
from ..losses import binary_cross_entropy
from ..metrics import accuracy


class LogisticRegression:

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        threshold: float = 0.5,
        loss="binary_cross_entropy",
        verbose=True,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold

        self.w = None
        self.b = None

        self.cost_history = []

        self.is_fitted = False

        self.verbose = verbose

    def _initialize_parameters(self, n_features: int):
        self.w = zeros((n_features, 1))
        self.b = 0.0

    def _forward(self, X: np.ndarray):
        z = np.dot(self.w.T, X) + self.b
        A = sigmoid(z)

        return A

    def _compute_loss(self, Y: np.ndarray, A: np.ndarray):
        return binary_cross_entropy(Y, A)

    def _backward(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        A: np.ndarray,
    ):
        m = X.shape[1]

        dZ = A - Y
        dw = (1 / m) * np.dot(X, dZ.T)
        db = (1 / m) * np.sum(dZ)

        return dw, db

    def _update_parameters(self, dw: np.ndarray, db: float):
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def fit(self, X: np.ndarray, Y: np.ndarray):
        n_features = X.shape[0]

        self._initialize_parameters(n_features)

        for epoch in range(self.epochs):
            # Forward propagation
            A = self._forward(X)

            # Compute loss
            cost = self._compute_loss(Y, A)

            # Backward propagation
            dw, db = self._backward(X, Y, A)

            # Gradient descent
            self._update_parameters(dw, db)

            # Save cost
            self.cost_history.append(cost)

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Cost: {cost:.6f}")

        self.is_fitted = True

        return self

    def predict_proba(self, X: np.ndarray):
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        return self._forward(X)

    def predict(self, X: np.ndarray):
        probability = self.predict_proba(X)
        predictions = (probability >= self.threshold).astype(int)

        return predictions

    def score(self, X: np.ndarray, Y: np.ndarray):
        predictions = self.predict(X)
        score = accuracy(Y, predictions)

        return score
