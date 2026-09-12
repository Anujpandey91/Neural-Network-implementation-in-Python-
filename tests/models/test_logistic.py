import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mydl.models import LogisticRegression


def load_dataset():

    data = load_breast_cancer()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.T
    X_test = X_test.T

    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(1, -1)

    return X_train, X_test, y_train, y_test


def test_fit():

    X_train, _, y_train, _ = load_dataset()

    model = LogisticRegression(
        learning_rate=0.01,
        epochs=100,
    )

    model.fit(X_train, y_train)

    assert model.is_fitted


def test_predict_proba():

    X_train, X_test, y_train, _ = load_dataset()

    model = LogisticRegression()

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)

    assert probs.shape == (1, X_test.shape[1])

    assert np.all(probs >= 0)

    assert np.all(probs <= 1)


def test_predict():

    X_train, X_test, y_train, _ = load_dataset()

    model = LogisticRegression()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    assert pred.shape == (1, X_test.shape[1])

    assert np.all(np.isin(pred, [0, 1]))


def test_score():

    X_train, X_test, y_train, y_test = load_dataset()

    model = LogisticRegression()

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    assert 0 <= score <= 1


def test_cost_decreases():

    X_train, _, y_train, _ = load_dataset()

    model = LogisticRegression(
        learning_rate=0.01,
        epochs=200,
    )

    model.fit(X_train, y_train)

    assert model.cost_history[0] > model.cost_history[-1]


def test_predict_before_fit():

    model = LogisticRegression()

    X = np.random.randn(30, 5)

    try:
        model.predict(X)
        assert False
    except ValueError:
        assert True
