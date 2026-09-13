import numpy as np

from ..initializers import zeros
from ..activations import sigmoid
from ..losses import binary_cross_entropy
from ..metrics import accuracy


class LogisticRegression:
    """
    Logistic regression model for binary classification.

    The model uses sigmoid activation to produce class probabilities and
    binary cross-entropy as the loss function. Parameters are optimized
    using gradient descent.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size used during gradient descent updates.
    epochs : int, default=1000
        Number of training iterations.
    threshold : float, default=0.5
        Probability threshold used to convert probabilities into
        binary class predictions.
    loss : str, default="binary_cross_entropy"
        Loss function used for training.
    verbose : bool, default=True
        Whether to print training progress during fitting.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        threshold: float = 0.5,
        loss: str = "binary_cross_entropy",
        verbose: bool = True,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold

        self.w = None
        self.b = None

        self.cost_history = []

        self.is_fitted = False

        self.verbose = verbose

    def _initialize_parameters(self, n_features: int) -> None:
        """Initialize model weights and bias to zero."""
        self.w = zeros((n_features, 1))
        self.b = 0.0

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation.

        Parameters
        ----------
        X : np.ndarray
            Input features with shape (n_features, n_examples).

        Returns
        -------
        np.ndarray
            Predicted probabilities with shape (1, n_examples).
        """
        z = np.dot(self.w.T, X) + self.b
        A = sigmoid(z)

        return A

    def _compute_loss(
        self,
        Y: np.ndarray,
        A: np.ndarray,
    ) -> float:
        """
        Compute binary cross-entropy loss.

        Parameters
        ----------
        Y : np.ndarray
            True binary labels with shape (1, n_examples).
        A : np.ndarray
            Predicted probabilities with shape (1, n_examples).

        Returns
        -------
        float
            Binary cross-entropy loss.
        """
        return binary_cross_entropy(Y, A)

    def _backward(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        A: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Compute gradients using backpropagation.

        Parameters
        ----------
        X : np.ndarray
            Input features with shape (n_features, n_examples).
        Y : np.ndarray
            True binary labels with shape (1, n_examples).
        A : np.ndarray
            Predicted probabilities with shape (1, n_examples).

        Returns
        -------
        tuple[np.ndarray, float]
            Gradients for the weights and bias, respectively.
        """
        m = X.shape[1]

        dZ = A - Y
        dw = (1 / m) * np.dot(X, dZ.T)
        db = (1 / m) * np.sum(dZ)

        return dw, db

    def _update_parameters(
        self,
        dw: np.ndarray,
        db: float,
    ) -> None:
        """Update weights and bias using gradient descent."""
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> "LogisticRegression":
        """
        Train the logistic regression model.

        Parameters
        ----------
        X : np.ndarray
            Training features with shape (n_features, n_examples).
        Y : np.ndarray
            Binary training labels with shape (1, n_examples).

        Returns
        -------
        LogisticRegression
            The fitted model instance.
        """
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
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

        return self._forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
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
