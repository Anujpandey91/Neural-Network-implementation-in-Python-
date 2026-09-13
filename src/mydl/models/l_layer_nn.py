import numpy as np

from ..metrics import accuracy
from ..initializers import random, zeros
from ..activations import relu, sigmoid, sigmoid_backward, relu_backward
from ..losses import binary_cross_entropy, binary_cross_entropy_backward
from ..layers import (
    linear_activation_backward,
    linear_activation_forward,
    Dropout,
)


class LLayerNN:
    """
    Fully connected feedforward neural network for binary classification.

    The network supports configurable hidden layers, pluggable optimizers,
    optional L1/L2 regularization, dropout, mini-batch training, and
    validation tracking.

    Parameters
    ----------
    hidden_layer : list[int]
        Number of neurons in each hidden layer.
    optimizer
        Optimizer instance used to update model parameters.
    regularizer, optional
        Regularization strategy used to penalize model weights.
    dropout, optional
        Dropout instance applied to hidden-layer activations during training.
    epochs : int, default=100
        Number of training epochs.
    threshold : float, default=0.5
        Probability threshold used to convert predictions into class labels.
    verbose : bool, default=True
        Whether to print training progress during fitting.
    """

    def __init__(
        self,
        hidden_layer: list[int],
        optimizer,
        regularizer=None,
        dropout=None,
        epochs: int = 100,
        threshold: float = 0.5,
        verbose: bool = True,
    ):
        self.hidden_layer = hidden_layer
        self.epochs = epochs
        self.threshold = threshold
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.dropout = dropout
        self.history = {"loss": [], "accuracy": []}
        self.is_fitted = False
        self.verbose = verbose
        self.parameters = {}
        self.grads = {}

    def _initialize_parameters(self) -> None:
        """Initialize network weights and biases using He initialization."""
        for l in range(1, len(self.layer_dims)):
            self.parameters["W" + str(l)] = random(
                (self.layer_dims[l], self.layer_dims[l - 1]),
                np.sqrt(2.0 / self.layer_dims[l - 1]),
            )
            self.parameters["b" + str(l)] = zeros((self.layer_dims[l], 1))

    def _forward(
        self,
        X: np.ndarray,
        training: bool = True,
    ) -> tuple[np.ndarray, list]:
        """
        Perform forward propagation through the network.

        Parameters
        ----------
        X : np.ndarray
            Input data with shape (n_features, n_examples).
        training : bool, default=True
            If True, dropout is applied to hidden-layer activations.

        Returns
        -------
        np.ndarray
            Output probabilities with shape (1, n_examples).
        list
            Cached values required for backpropagation.
        """
        A = X
        L = int(len(self.parameters) / 2)
        caches = []

        for l in range(1, L):
            W_l = self.parameters["W" + str(l)]
            b_l = self.parameters["b" + str(l)]

            A_i, cache = linear_activation_forward(
                A,
                W_l,
                b_l,
                relu,
            )

            if self.dropout is not None:
                A_i, dropout_mask = self.dropout.forward(
                    A_i,
                    training=training,
                )
            else:
                dropout_mask = None

            A = A_i
            caches.append((cache, dropout_mask))

        W = self.parameters["W" + str(L)]
        b = self.parameters["b" + str(L)]

        AL, cache = linear_activation_forward(
            A,
            W,
            b,
            sigmoid,
        )

        caches.append(cache)

        return AL, caches

    def _compute_loss(
        self,
        Y: np.ndarray,
        A: np.ndarray,
        m: int,
    ) -> float:
        """Compute binary cross-entropy loss with optional regularization."""
        data_loss = binary_cross_entropy(Y, A)

        if self.regularizer is None:
            return data_loss

        regularization_loss = self.regularizer.penalty(
            self.parameters,
            m,
        )

        return data_loss + regularization_loss

    def _backward(
        self,
        Y: np.ndarray,
        A: np.ndarray,
        caches: list,
        m: int,
    ) -> None:
        """Perform backpropagation and compute gradients for all parameters."""
        L = int(len(self.parameters) / 2)

        # Output layer: Linear -> Sigmoid
        dAL = binary_cross_entropy_backward(Y, A)

        dA_prev, dWL, dbL = linear_activation_backward(
            dAL,
            caches[L - 1],
            sigmoid_backward,
        )

        if self.regularizer is not None:
            dWL = dWL + self.regularizer.gradient(
                self.parameters["W" + str(L)],
                m,
            )

        self.grads["dW" + str(L)] = dWL
        self.grads["db" + str(L)] = dbL

        # Hidden layers: Linear -> ReLU -> Dropout
        for l in range(L - 1, 0, -1):
            cache, dropout_mask = caches[l - 1]

            if self.dropout is not None:
                dA_prev = self.dropout.backward(
                    dA_prev,
                    dropout_mask,
                )

            dA_prev, dWl, dbl = linear_activation_backward(
                dA_prev,
                cache,
                relu_backward,
            )

            if self.regularizer is not None:
                dWl = dWl + self.regularizer.gradient(
                    self.parameters["W" + str(l)],
                    m,
                )

            self.grads["dW" + str(l)] = dWl
            self.grads["db" + str(l)] = dbl

    def _update_parameters(self) -> None:
        """Update model parameters using the configured optimizer."""
        self.optimizer.update(
            self.parameters,
            self.grads,
        )

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int | None = None,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "LLayerNN":
        """
        Train the neural network on labeled training data.

        Parameters
        ----------
        X : np.ndarray
            Training features with shape (n_features, n_examples).
        Y : np.ndarray
            Binary training labels with shape (1, n_examples).
        batch_size : int, optional
            Number of examples processed per optimization step.
            If None, the full training set is used for each update.
        validation_data : tuple[np.ndarray, np.ndarray], optional
            Validation data provided as (X_val, Y_val).

        Returns
        -------
        LLayerNN
            The fitted model instance.

        Raises
        ------
        TypeError
            If batch_size is not an integer.
        ValueError
            If batch_size is not positive or validation_data has an
            invalid structure or incompatible dimensions.
        """
        # 1. Get dimensions
        n_features, m = X.shape

        # 2. Validate batch_size
        if batch_size is None:
            batch_size = m
        elif not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        elif batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        else:
            batch_size = min(batch_size, m)

        # 3. Validate validation data
        if validation_data is not None:
            if not isinstance(validation_data, tuple) or len(validation_data) != 2:
                raise ValueError("validation_data must be a tuple of (X_val, Y_val)")

            X_val, Y_val = validation_data

            if X_val.shape[0] != n_features:
                raise ValueError("X_val must have the same number of features as X")

            if X_val.shape[1] != Y_val.shape[1]:
                raise ValueError(
                    "X_val and Y_val must contain the same number of examples"
                )

        # 4. Set up layer dimensions
        self.layer_dims = [
            n_features,
            *self.hidden_layer,
            1,
        ]

        # 5. Initialize parameters
        self._initialize_parameters()

        # Reset history
        self.history = {
            "loss": [],
            "accuracy": [],
        }

        if validation_data is not None:
            self.history["val_loss"] = []
            self.history["val_accuracy"] = []

        # 6. Training
        for epoch in range(self.epochs):
            # 6a. Shuffle the dataset
            permutation = np.random.permutation(m)

            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]

            # 6b. Track total cost for this epoch
            epoch_cost = 0
            num_batches = 0

            # 6c. Go through mini-batches
            for start in range(0, m, batch_size):
                end = start + batch_size

                X_batch = X_shuffled[:, start:end]
                Y_batch = Y_shuffled[:, start:end]

                # Forward propagation
                prediction, caches = self._forward(
                    X_batch,
                    training=True,
                )

                # Compute loss
                batch_cost = self._compute_loss(
                    Y_batch,
                    prediction,
                    m,
                )

                # Backpropagation
                self._backward(
                    Y_batch,
                    prediction,
                    caches,
                    m,
                )

                # Update parameters
                self._update_parameters()

                epoch_cost += batch_cost
                num_batches += 1

            # 6d. Calculate epoch cost
            cost = epoch_cost / num_batches

            # 6e. Calculate training accuracy
            epoch_predictions, _ = self._forward(
                X,
                training=False,
            )

            epoch_predictions = (epoch_predictions >= self.threshold).astype(int)

            epoch_accuracy = np.mean(epoch_predictions == Y)

            # 6f. Calculate validation metrics
            if validation_data is not None:
                val_prediction, _ = self._forward(
                    X_val,
                    training=False,
                )

                val_loss = self._compute_loss(
                    Y_val,
                    val_prediction,
                    X_val.shape[1],
                )

                val_predictions = (val_prediction >= self.threshold).astype(int)

                val_accuracy = np.mean(val_predictions == Y_val)

            # 6g. Store history
            self.history["loss"].append(cost)
            self.history["accuracy"].append(epoch_accuracy)

            if validation_data is not None:
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_accuracy)

            # Print training progress
            if self.verbose and epoch % 100 == 0:
                if validation_data is not None:
                    print(
                        f"Epoch {epoch:4d} | "
                        f"Loss: {cost:.6f} | "
                        f"Accuracy: {epoch_accuracy:.4f} | "
                        f"Val Loss: {val_loss:.6f} | "
                        f"Val Accuracy: {val_accuracy:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch:4d} | "
                        f"Loss: {cost:.6f} | "
                        f"Accuracy: {epoch_accuracy:.4f}"
                    )

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
            raise ValueError("Model has not been fitted.")

        A, _ = self._forward(
            X,
            training=False,
        )

        return A

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
