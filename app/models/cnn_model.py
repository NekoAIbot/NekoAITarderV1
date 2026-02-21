import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "cnn_model.keras"

class CNNModel(BaseEstimator, ClassifierMixin):

    def __init__(self, lookback=20, epochs=20, batch_size=32):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    def get_model(self):
        return self

    def _build_model(self, n_features):
        model = Sequential([
            Input(shape=(self.lookback, n_features)),
            Conv1D(32, 3, activation="relu"),
            MaxPool1D(2),
            Dropout(0.2),
            Conv1D(64, 3, activation="relu"),
            MaxPool1D(2),
            Dropout(0.2),
            Flatten(),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def _reshape(self, X):
        """
        Reshape input to [samples, lookback, features], including multi-window sentiment.
        Automatically handles all columns (technical + sentiment).
        """
        arr = X.values if hasattr(X, "values") else X
        n_samples = len(arr) - self.lookback
        if n_samples <= 0:
            return np.zeros((0, self.lookback, arr.shape[1]))
        sequences = np.array([arr[i:i+self.lookback] for i in range(n_samples)])
        return sequences

    def fit(self, X, y):
        X_seq = self._reshape(X)
        y_seq = y[self.lookback:]

        if len(X_seq) == 0:
            raise ValueError("Not enough samples for CNN.")

        self.model = self._build_model(X.shape[1])
        es = EarlyStopping(patience=5, restore_best_weights=True)
        self.model.fit(
            X_seq,
            y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[es],
            verbose=0
        )

        self.classes_ = np.unique(y)
        self.model.save(MODEL_FILE)
        return self

    def predict_proba(self, X):
        X_seq = self._reshape(X)
        if len(X_seq) == 0:
            return np.zeros((len(X), 2))

        preds = self.model.predict(X_seq, verbose=0).flatten()
        padded = np.concatenate([np.zeros(self.lookback), preds])
        return np.column_stack([1 - padded, padded])

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

__all__ = ["CNNModel"]