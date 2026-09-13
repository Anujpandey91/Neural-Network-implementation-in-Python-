import numpy as np

from ..initializers import zeros, random
from ..layers import (
    linear_activation_forward,
    linear_activation_backward,
)
from ..activations import (
    relu,
    sigmoid,
    sigmoid_backward,
    relu_backward,
)
from ..losses import (
    binary_cross_entropy,
    binary_cross_entropy_backward,
)
from ..metrics import accuracy


class TwoLayerNN:
    """
    Two-layer fully connected neural network for binary classification.

    The network consists of one hidden layer with ReLU activation and
    an output layer with sigmoid activation. Parameters are optimized
    using gradient descent and binary cross-entropy loss.

    Parameters
    ----------
    hidden_units : int, default=16
        Number of neurons in the hidden layer.
    learning_rate : float, default=0.01
        Step size used during gradient descent updates.
    epochs : int, default=100
        Number of training iterations.
    threshold : float, default=0.5
        Probability threshold used to convert probabilities into
        binary class predictions.
    verbose : bool, default=True
        Whether to print training progress during fitting.
    """

    def __init__(
        self,
        hidden_units: int = 16,
        learning_rate: float = 0.01,
        epochs: int = 100,
        threshold: float = 0.5,
        verbose: bool = True,
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

    def _initialize_parameters(self, n_features: int) -> None:
        """Initialize weights using He initialization and biases to zero."""
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

    def _forward(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, tuple]:
        """
        Perform forward propagation through the network.

        Parameters
        ----------
        X : np.ndarray
            Input features with shape (n_features, n_examples).

        Returns
        -------
        np.ndarray
            Predicted probabilities with shape (1, n_examples).
        tuple
            Cached values required for backpropagation.
        """
        A1, cache1 = linear_activation_forward(
            X,
            self.W1,
            self.b1,
            relu,
        )

        A2, cache2 = linear_activation_forward(
            A1,
            self.W2,
            self.b2,
            sigmoid,
        )

        caches = (cache1, cache2)
        self.A2 = A2

        return A2, caches

    def _compute_loss(
        self,
        Y: np.ndarray,
        A2: np.ndarray,
    ) -> float:
        """Compute binary cross-entropy loss."""
        return binary_cross_entropy(Y, A2)

    def _backward(
        self,
        Y: np.ndarray,
        A2: np.ndarray,
        caches: tuple,
    ) -> None:
        """Compute gradients using backpropagation."""
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

    def _update_parameters(self) -> None:
        """Update weights and biases using gradient descent."""
        self.W1 = self.W1 - self.learning_rate * self.dW1
        self.b1 = self.b1 - self.learning_rate * self.db1

        self.W2 = self.W2 - self.learning_rate * self.dW2
        self.b2 = self.b2 - self.learning_rate * self.db2

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> "TwoLayerNN":
        """
        Train the neural network on labeled training data.

        Parameters
        ----------
        X : np.ndarray
            Training features with shape (n_features, n_examples).
        Y : np.ndarray
            Binary training labels with shape (1, n_examples).

        Returns
        -------
        TwoLayerNN
            The fitted model instance.
        """
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

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class probabilities for input examples.

        Parameters
        ----------
        X : np.ndarray
            Input features with shape (n_features, n_examples).

        Returns
        -------
        np.ndarray
            Predicted probabilities with shape (1, n_examples).

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        A2, _ = self._forward(X)

        return A2

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict binary class labels for input examples.

        Parameters
        ----------
        X : np.ndarray
            Input features with shape (n_features, n_examples).

        Returns
        -------
        np.ndarray
            Predicted binary labels with shape (1, n_examples).
        """
        probability = self.predict_proba(X)

        predictions = (probability >= self.threshold).astype(int)

        return predictions

    def score(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> float:
        """
        Compute classification accuracy on a dataset.

        Parameters
        ----------
        X : np.ndarray
            Input features with shape (n_features, n_examples).
        Y : np.ndarray
            True binary labels with shape (1, n_examples).

        Returns
        -------
        float
            Classification accuracy between 0.0 and 1.0.
        """
        predictions = self.predict(X)

        score = accuracy(
            Y,
            predictions,
        )

        return score
