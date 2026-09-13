import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mydl.models import LLayerNN
from mydl.optimizers import Momentum
from mydl.optimizers import RMSProp


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


class TestLLayerNN:

    @classmethod
    def setup_class(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = load_dataset()

    def test_fit(self):
        model = LLayerNN(
            hidden_layer=[32, 16, 8],
            optimizer=Momentum(learning_rate=0.01),
            epochs=100,
            verbose=False,
        )

        model.fit(self.X_train, self.y_train)

        assert model.is_fitted

    def test_cost_decreases(self):
        model = LLayerNN(
            hidden_layer=[32, 16, 8],
            optimizer=RMSProp(learning_rate=0.01),
            epochs=100,
            verbose=False,
        )

        model.fit(self.X_train, self.y_train)

        assert model.history["loss"][-1] < model.history["loss"][0]

    def test_training_history(self):
        epochs = 10

        model = LLayerNN(
            hidden_layer=[16, 8],
            optimizer=Momentum(learning_rate=0.01),
            epochs=epochs,
            verbose=False,
        )

        model.fit(self.X_train, self.y_train)

        assert set(model.history.keys()) == {"loss", "accuracy"}
        assert len(model.history["loss"]) == epochs
        assert len(model.history["accuracy"]) == epochs

    def test_validation_history(self):
        epochs = 10

        model = LLayerNN(
            hidden_layer=[16, 8],
            optimizer=Momentum(learning_rate=0.01),
            epochs=epochs,
            verbose=False,
        )

        model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
        )

        assert set(model.history.keys()) == {
            "loss",
            "accuracy",
            "val_loss",
            "val_accuracy",
        }

        assert len(model.history["loss"]) == epochs
        assert len(model.history["accuracy"]) == epochs
        assert len(model.history["val_loss"]) == epochs
        assert len(model.history["val_accuracy"]) == epochs

    def test_invalid_validation_data_type(self):
        model = LLayerNN(
            hidden_layer=[16, 8],
            optimizer=Momentum(learning_rate=0.01),
            epochs=1,
            verbose=False,
        )

        with pytest.raises(
            ValueError,
            match="validation_data must be a tuple",
        ):
            model.fit(
                self.X_train,
                self.y_train,
                validation_data=self.X_test,
            )

    def test_invalid_validation_data_length(self):
        model = LLayerNN(
            hidden_layer=[16, 8],
            optimizer=Momentum(learning_rate=0.01),
            epochs=1,
            verbose=False,
        )

        with pytest.raises(
            ValueError,
            match="validation_data must be a tuple",
        ):
            model.fit(
                self.X_train,
                self.y_train,
                validation_data=(self.X_test,),
            )

    def test_validation_feature_mismatch(self):
        model = LLayerNN(
            hidden_layer=[16, 8],
            optimizer=Momentum(learning_rate=0.01),
            epochs=1,
            verbose=False,
        )

        X_val = self.X_test[:-1, :]

        with pytest.raises(
            ValueError,
            match="X_val must have the same number of features as X",
        ):
            model.fit(
                self.X_train,
                self.y_train,
                validation_data=(X_val, self.y_test),
            )

    def test_validation_example_count_mismatch(self):
        model = LLayerNN(
            hidden_layer=[16, 8],
            optimizer=Momentum(learning_rate=0.01),
            epochs=1,
            verbose=False,
        )

        Y_val = self.y_test[:, :-1]

        with pytest.raises(
            ValueError,
            match="X_val and Y_val must contain the same number of examples",
        ):
            model.fit(
                self.X_train,
                self.y_train,
                validation_data=(self.X_test, Y_val),
            )

    def test_predict_proba_before_fit(self):
        model = LLayerNN(
            hidden_layer=[16, 8],
            optimizer=Momentum(learning_rate=0.01),
            epochs=1,
            verbose=False,
        )

        with pytest.raises(
            ValueError,
            match="Model has not been fitted.",
        ):
            model.predict_proba(self.X_test)

    def test_predict_before_fit(self):
        model = LLayerNN(
            hidden_layer=[16, 8],
            optimizer=Momentum(learning_rate=0.01),
            epochs=1,
            verbose=False,
        )

        with pytest.raises(
            ValueError,
            match="Model has not been fitted.",
        ):
            model.predict(self.X_test)

    def test_score_before_fit(self):
        model = LLayerNN(
            hidden_layer=[16, 8],
            optimizer=Momentum(learning_rate=0.01),
            epochs=1,
            verbose=False,
        )

        with pytest.raises(
            ValueError,
            match="Model has not been fitted.",
        ):
            model.score(self.X_test, self.y_test)

    def test_predict_proba_shape(self):
        model = LLayerNN(
            hidden_layer=[32, 16, 8],
            optimizer=Momentum(learning_rate=0.01),
            epochs=100,
            verbose=False,
        )

        model.fit(self.X_train, self.y_train)

        prob = model.predict_proba(self.X_test)

        assert prob.shape == self.y_test.shape

    def test_probability_range(self):
        model = LLayerNN(
            hidden_layer=[32, 16, 8],
            optimizer=Momentum(learning_rate=0.01),
            epochs=100,
            verbose=False,
        )

        model.fit(self.X_train, self.y_train)

        prob = model.predict_proba(self.X_test)

        assert np.all(prob >= 0)
        assert np.all(prob <= 1)

    def test_predict_shape(self):
        model = LLayerNN(
            hidden_layer=[32, 16, 8],
            optimizer=Momentum(learning_rate=0.01),
            epochs=100,
            verbose=False,
        )

        model.fit(self.X_train, self.y_train)

        pred = model.predict(self.X_test)

        assert pred.shape == self.y_test.shape

    def test_score(self):
        model = LLayerNN(
            hidden_layer=[32, 16, 8],
            optimizer=Momentum(learning_rate=0.01),
            epochs=1000,
            verbose=False,
        )

        model.fit(self.X_train, self.y_train)

        score = model.score(self.X_test, self.y_test)

        assert score > 0.90
