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

    def __init__(
        self,
        hidden_layer: list[int],
        optimizer,
        regularizer=None,
        dropout=None,
        epochs: int = 100,
        threshold: float = 0.5,
        verbose=True,
    ):
        self.hidden_layer = hidden_layer
        self.epochs = epochs
        self.threshold = threshold
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.dropout = dropout

        self.cost_history = []
        self.is_fitted = False
        self.verbose = verbose

        self.parameters = {}
        self.grads = {}

    def _initialize_parameters(self):
        for l in range(1, len(self.layer_dims)):
            self.parameters["W" + str(l)] = random(
                (self.layer_dims[l], self.layer_dims[l - 1]),
                np.sqrt(2.0 / self.layer_dims[l - 1]),
            )
            self.parameters["b" + str(l)] = zeros((self.layer_dims[l], 1))

    def _forward(self, X, training=True):
        A = X
        L = int(len(self.parameters) / 2)
        caches = []

        for l in range(1, L):
            W_l = self.parameters["W" + str(l)]
            b_l = self.parameters["b" + str(l)]

            A_i, cache = linear_activation_forward(
                A, W_l, b_l, relu
            )

            if self.dropout is not None:
                A_i, dropout_mask = self.dropout.forward(
                    A_i,
                    training=training
                )
            else:
                dropout_mask = None

            A = A_i
            caches.append((cache, dropout_mask))

        W = self.parameters["W" + str(L)]
        b = self.parameters["b" + str(L)]
        AL, cache = linear_activation_forward(A, W, b, sigmoid)
        caches.append(cache)

        return AL, caches

    def _compute_loss(self, Y, A, m):

        data_loss = binary_cross_entropy(Y, A)

        if self.regularizer is None:
            return data_loss

        regularization_loss = self.regularizer.penalty(self.parameters, m)
        return data_loss + regularization_loss


    def _backward(self, Y, A, caches, m):
        L = int(len(self.parameters) / 2)

        # Output layer: Linear -> Sigmoid
        dAL = binary_cross_entropy_backward(Y, A)

        dA_prev, dWL, dbL = linear_activation_backward(
            dAL,
            caches[L - 1],
            sigmoid_backward,
        )

        if self.regularizer is not None:
            dWL = dWL + self.regularizer.gradient(self.parameters["W" + str(L)], m)

        self.grads["dW" + str(L)] = dWL
        self.grads["db" + str(L)] = dbL

        # Hidden layers: Linear -> ReLU -> Dropout
        for l in range(L - 1, 0, -1):

            cache, dropout_mask = caches[l - 1]

            if self.dropout is not None:
                dA_prev = self.dropout.backward(dA_prev, dropout_mask)

            dA_prev, dWl, dbl = linear_activation_backward(
                dA_prev,
                cache,
                relu_backward,
            )

            if self.regularizer is not None:
                dWl = dWl + self.regularizer.gradient(self.parameters["W" + str(l)], m)

            self.grads["dW" + str(l)] = dWl
            self.grads["db" + str(l)] = dbl


    def _update_parameters(self):
        self.optimizer.update(self.parameters, self.grads)


    def fit(self, X: np.ndarray, Y: np.ndarray, batch_size=None):

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

        # 3. Set up layer dimensions
        self.layer_dims = [
            n_features,
            *self.hidden_layer,
            1,
        ]

        # 4. Initialize parameters
        self._initialize_parameters()

        # 5. Training
        for epoch in range(self.epochs):
            # 5a. Shuffle the dataset
            permutation = np.random.permutation(m)

            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]

            # 5b. Track total cost for this epoch
            epoch_cost = 0
            num_batches = 0

            # 5c. Go through mini-batches
            for start in range(0, m, batch_size):

                # determine end
                end = start + batch_size
                # get X_batch
                X_batch = X_shuffled[:, start:end]
                # get Y_batch
                Y_batch = Y_shuffled[:, start:end]

                # forward
                prediction, caches = self._forward(X_batch, training=True)

                # loss
                batch_cost = self._compute_loss(Y_batch, prediction, m)

                # backward
                self._backward(Y_batch, prediction, caches, m)

                # update
                self._update_parameters()

                # accumulate cost
                epoch_cost += batch_cost
                num_batches += 1

            # 5d. Calculate epoch cost
            cost = epoch_cost / num_batches

            # 5e. Store epoch cost
            self.cost_history.append(cost)

            # 5f. Print
            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Cost: {cost:.6f}")

        self.is_fitted = True

        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model has not been fitted.")

        A, _ = self._forward(X, training=False)

        return A

    def predict(self, X):
        probability = self.predict_proba(X)
        predictions = (probability >= self.threshold).astype(int)

        return predictions

    def score(self, X, Y):
        predictions = self.predict(X)
        score = accuracy(Y, predictions)

        return score
