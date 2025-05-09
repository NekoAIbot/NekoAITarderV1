import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

MODEL_DIR = Path(__file__).parent / "models"

class BaseNN:
    def __init__(self, name, build_fn):
        """
        build_fn -> returns a compiled Keras model
        """
        self.name = name
        self.build_fn = build_fn
        self.lookback = 20
        MODEL_DIR.mkdir(exist_ok=True)

    def _windowed_data(self, df, news_series):
        Xs, ys = [], []
        feat_df = self.featurize(df, news_series)
        arr = feat_df.values
        for i in range(self.lookback, len(arr)-1):
            Xs.append(arr[i-self.lookback+1:i+1])
            ys.append(1 if df["close"].iloc[i+1]>df["close"].iloc[i] else 0)
        return np.array(Xs), np.array(ys)

    @staticmethod
    def featurize(df, news):
        # flatten features: close,high,low,open,volume + news
        data = df[["open","high","low","close","volume"]].copy()
        data["sent"] = news
        return data.dropna()

    def fit(self, df, news_series):
        X, y = self._windowed_data(df, news_series)
        model = self.build_fn(input_shape=X.shape[1:])
        es = EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1,
                  callbacks=[es], verbose=0)
        model.save(MODEL_DIR/self.name)
        self.model = model

    def predict(self, df, news):
        feat_df = self.featurize(df.tail(self.lookback), [news]*self.lookback)
        X = feat_df.values.reshape(1, self.lookback, -1)
        p = self.model.predict(X)[0,0]
        sig = "BUY" if p>0.5 else "SELL"
        return {"signal":sig, "confidence":max(p,1-p)*100, "predicted_change":(p-(1-p))*100}


def build_dense(input_shape):
    m = Sequential([
        Flatten(input_shape=input_shape),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    m.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return m

def build_cnn(input_shape):
    m = Sequential([
        Conv1D(32, 3, activation="relu", input_shape=input_shape),
        MaxPooling1D(2),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    m.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return m

def build_lstm(input_shape):
    m = Sequential([
        LSTM(32, input_shape=input_shape),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    m.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return m

# instantiate your wrappers
DenseModel = lambda: BaseNN("dense", build_dense)
CNNModel   = lambda: BaseNN("cnn",   build_cnn)
LSTMModel  = lambda: BaseNN("lstm",  build_lstm)
