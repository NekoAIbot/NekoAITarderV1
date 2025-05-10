import numpy as np, pandas as pd
from pathlib import Path
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout

MODEL_DIR  = Path(__file__).parent/"models"
MODEL_FILE = MODEL_DIR/"dense_model.h5"

def build_features(df):
    # re-use XGBModel.featurize under the hood:
    from .xgb_model import MomentumModel as Feat
    return Feat.featurize(df, pd.Series(0, index=df.index))  # placeholder

class DenseNNModel:
    def __init__(self, lookback=20):
        self.lookback = lookback
        MODEL_DIR.mkdir(exist_ok=True)
        if MODEL_FILE.exists():
            self.model = load_model(MODEL_FILE)
        else:
            m = Sequential([
                Flatten(input_shape=(lookback,13)),
                Dense(128, activation="relu"), Dropout(0.2),
                Dense(64,  activation="relu"), Dropout(0.2),
                Dense(1, activation="sigmoid")
            ])
            m.compile("adam","binary_crossentropy",metrics=["accuracy"])
            self.model = m

    def _prepare(self, df, news):
        feat = build_features(df)
        feat["sentiment"] = news
        X,y = [],[]
        for i in range(self.lookback, len(feat)-1):
            X.append(feat.iloc[i-self.lookback:i].values)
            y.append(int(feat["sentiment"].iloc[i+1]>feat["sentiment"].iloc[i]))
        return np.array(X), np.array(y)

    def fit(self, df, news):
        X,y = self._prepare(df, news)
        self.model.fit(X,y,validation_split=0.2,epochs=10,batch_size=32,verbose=0)
        self.model.save(MODEL_FILE)

    def predict(self, df, news):
        X,_ = self._prepare(df, pd.Series(news,index=df.index))
        if len(X)==0:
            return {"signal":"HOLD","confidence":0.0,"predicted_change":0.0}
        p = float(self.model.predict(X[[-1]])[0][0])
        sig = "BUY" if p>0.5 else "SELL"
        return {"signal":sig,"confidence":p*100,"predicted_change":(p-0.5)*200}
