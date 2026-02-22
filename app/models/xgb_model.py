import joblib
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "xgb_model.joblib"


class MomentumModel(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=1.0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        self.pipeline = None
        self._build_pipeline()

    def get_model(self):
        return self

    def _build_pipeline(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                eval_metric="logloss",
                scale_pos_weight=self.scale_pos_weight,
                tree_method="hist"
            )),
        ])

    def fit(self, X, y):
        self._build_pipeline()
        self.pipeline.fit(X, y)
        self.classes_ = np.unique(y)
        joblib.dump(self.pipeline, MODEL_FILE)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)