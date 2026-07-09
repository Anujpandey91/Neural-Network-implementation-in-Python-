"""
Logistic Regression Example

This example demonstrates how to train and evaluate a Logistic Regression
model using the mydl library on the Breast Cancer dataset.
"""

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

from mydl.models import LogisticRegression


def load_data():
    """Load the Breast Cancer dataset."""

    data = load_breast_cancer()

    X = data.data
    y = data.target

    return X, y


def preprocess_data(X, y):
    """
    Split the dataset and preprocess it for mydl.

    Returns
    -------
    X_train, X_test : np.ndarray
        Shape (n_features, m)

    y_train, y_test : np.ndarray
        Shape (1, m)
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to mydl format
    X_train = X_train.T
    X_test = X_test.T

    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(1, -1)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train Logistic Regression."""

    model = LogisticRegression(
        learning_rate=0.01,
        epochs=5000,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""

    accuracy = model.score(X_test, y_test)

    print("=" * 50)
    print("MyDL Logistic Regression")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.4f}")

    return accuracy


def compare_with_sklearn(X_train, y_train, X_test, y_test):
    """Compare against scikit-learn."""

    sklearn_model = SkLogisticRegression(
        max_iter=5000,
    )

    sklearn_model.fit(
        X_train.T,
        y_train.ravel(),
    )

    accuracy = sklearn_model.score(
        X_test.T,
        y_test.ravel(),
    )

    print("=" * 50)
    print("Scikit-Learn Logistic Regression")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.4f}")

    return accuracy


def plot_learning_curve(model):
    """Plot the training loss."""

    plt.figure(figsize=(8, 5))

    plt.plot(model.cost_history)

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Binary Cross Entropy Loss")

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def main():

    # Load dataset
    X, y = load_data()

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Compare
    compare_with_sklearn(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # Plot
    plot_learning_curve(model)


if __name__ == "__main__":
    main()
