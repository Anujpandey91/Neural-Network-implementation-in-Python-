import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from mydl.models import LLayerNN


def load_data():
    """
    Load the Breast Cancer dataset.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    y : np.ndarray
        Target labels of shape (n_samples,).
    """
    data = load_breast_cancer()
    return data.data, data.target


def preprocess_data(X, y):
    """
    Split and preprocess the dataset for mydl.

    Returns
    -------
    X_train : np.ndarray
        Shape (n_features, m_train)

    X_test : np.ndarray
        Shape (n_features, m_test)

    y_train : np.ndarray
        Shape (1, m_train)

    y_test : np.ndarray
        Shape (1, m_test)
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
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
    """
    Train the mydl deep neural network.
    """

    model = LLayerNN(
        hidden_layer=[32, 16, 8],
        learning_rate=0.01,
        epochs=1000,
        verbose=True,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the trained mydl model.
    """

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print("=" * 60)
    print("MyDL Deep Neural Network")
    print("=" * 60)
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print("=" * 60)


def compare_with_sklearn(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Compare mydl with sklearn's MLPClassifier.
    """

    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16, 8),
        activation="relu",
        learning_rate_init=0.01,
        max_iter=500,
        random_state=42,
    )

    clf.fit(X_train.T, y_train.ravel())

    score = clf.score(X_test.T, y_test.ravel())

    print("\nScikit-Learn MLPClassifier")
    print("-" * 60)
    print(f"Test Accuracy : {score:.4f}")


def plot_learning_curve(model):
    """
    Plot the training loss curve.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(model.cost_history, linewidth=2)

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
    evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # Compare with sklearn
    compare_with_sklearn(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # Plot learning curve
    plot_learning_curve(model)


if __name__ == "__main__":
    main()
