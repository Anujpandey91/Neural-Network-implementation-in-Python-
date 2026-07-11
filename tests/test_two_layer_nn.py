import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mydl.models import TwoLayerNN


def load_dataset():
    data = load_breast_cancer()

    X = data.data
    y = data.target

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

    X_train = X_train.T
    X_test = X_test.T

    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(1, -1)

    return X_train, X_test, y_train, y_test


class TestTwoLayerNN:

    @classmethod
    def setup_class(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = load_dataset()

    def test_fit(self):
        model = TwoLayerNN(
            hidden_units=16,
            learning_rate=0.01,
            epochs=100,
            verbose=False,
        )

        model.fit(self.X_train, self.y_train)

        assert model.is_fitted

    def test_cost_decreases(self):
        model = TwoLayerNN(
            hidden_units=16,
            learning_rate=0.01,
            epochs=100,
            verbose=False,
        )

        model.fit(self.X_train, self.y_train)

        assert model.cost_history[-1] < model.cost_history[0]

    def test_predict_proba_shape(self):
        model = TwoLayerNN(
            hidden_units=16,
            learning_rate=0.01,
            epochs=100,
            verbose=False,
        )

        model.fit(self.X_train, self.y_train)

        prob = model.predict_proba(self.X_test)

        assert prob.shape == self.y_test.shape

    def test_probability_range(self):
        model = TwoLayerNN(
            hidden_units=16,
            learning_rate=0.01,
            epochs=100,
            verbose=False,
        )

        model.fit(self.X_train, self.y_train)

        prob = model.predict_proba(self.X_test)

        assert np.all(prob >= 0)
        assert np.all(prob <= 1)

    def test_predict_shape(self):
        model = TwoLayerNN(
            hidden_units=16,
            learning_rate=0.01,
            epochs=100,
            verbose=False,
        )

        model.fit(self.X_train, self.y_train)

        pred = model.predict(self.X_test)

        assert pred.shape == self.y_test.shape

    def test_score(self):
        model = TwoLayerNN(
            hidden_units=16,
            learning_rate=0.01,
            epochs=200,
            verbose=False,
        )

        model.fit(self.X_train, self.y_train)

        score = model.score(self.X_test, self.y_test)

        assert score > 0.85
