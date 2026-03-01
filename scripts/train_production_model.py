#!/usr/bin/env python3
"""Production-oriented trainer for live trading signal model.

This script builds a richer multi-symbol + multi-timeframe dataset, performs a
small time-aware hyperparameter search, evaluates on holdout, and persists the
best pipeline to the runtime model location used by app/models/ai_model.py.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FOREX_MAJORS, CRYPTO_ASSETS
from app.market_data import fetch_market_data
from app.models.ai_model import MomentumModel

MODEL_PATH = ROOT / "app" / "models" / "models" / "xgb_model.joblib"
ALT_MODEL_PATH = ROOT / "models" / "xgb_model_production.joblib"
REPORT_PATH = ROOT / "models" / "xgb_model_training_report.json"

TIMEFRAMES = ["1m", "5m", "15m", "1h"]
LIMIT_BY_TF = {
    "1m": 3500,
    "5m": 3000,
    "15m": 2500,
    "1h": 2000,
}

PARAM_CANDIDATES = [
    {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.04, "subsample": 0.9, "colsample_bytree": 0.9},
    {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85},
    {"n_estimators": 350, "max_depth": 4, "learning_rate": 0.06, "subsample": 0.8, "colsample_bytree": 0.8},
]


def _build_dataset(symbols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    frames: list[pd.DataFrame] = []
    targets: list[pd.Series] = []

    for symbol in symbols:
        for timeframe in TIMEFRAMES:
            limit = LIMIT_BY_TF[timeframe]
            try:
                df = fetch_market_data(symbol, timeframe=timeframe, limit=limit)
            except Exception:
                continue

            if df is None or len(df) < 200:
                continue

            # Keep timestamp for chronological split later.
            df2 = df.copy()
            df2["_source_symbol"] = symbol
            df2["_source_tf"] = timeframe

            feat = MomentumModel.featurize(df2[["open", "high", "low", "close", "volume"]], news=0.0)
            if feat.empty:
                continue

            target = (df2.loc[feat.index, "close"].shift(-1) > df2.loc[feat.index, "close"]).astype(int)
            valid = target.notna()
            feat = feat.loc[valid].copy()
            target = target.loc[valid].astype(int)

            feat["_ts"] = pd.to_datetime(feat.index, utc=True, errors="coerce")
            feat["_source_symbol"] = symbol
            feat["_source_tf"] = timeframe

            frames.append(feat)
            targets.append(target)

    if not frames:
        raise RuntimeError("No training samples generated. Check data/provider availability.")

    X = pd.concat(frames, axis=0)
    y = pd.concat(targets, axis=0)
    return X, y


def _time_split(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8):
    Xs = X.copy()
    Xs["_y"] = y.values
    Xs = Xs.sort_values("_ts")

    split_idx = int(len(Xs) * train_ratio)
    train = Xs.iloc[:split_idx]
    test = Xs.iloc[split_idx:]

    drop_cols = ["_ts", "_source_symbol", "_source_tf", "_y"]
    X_train = train.drop(columns=drop_cols)
    y_train = train["_y"]
    X_test = test.drop(columns=drop_cols)
    y_test = test["_y"]

    return X_train, y_train, X_test, y_test


def _evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    out = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    if len(np.unique(y_test)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_test, proba))
    else:
        out["roc_auc"] = 0.5
    return out


def _make_pipeline(params: dict) -> Pipeline:
    clf = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=42,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def main() -> int:
    symbols = FOREX_MAJORS + CRYPTO_ASSETS
    print(f"Building dataset from {len(symbols)} symbols x {len(TIMEFRAMES)} timeframes...")
    X, y = _build_dataset(symbols)
    print(f"Dataset rows: {len(X)}")

    X_train, y_train, X_test, y_test = _time_split(X, y)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    best = None
    best_score = -1.0
    all_results = []

    for params in PARAM_CANDIDATES:
        pipe = _make_pipeline(params)
        pipe.fit(X_train, y_train)
        metrics = _evaluate(pipe, X_test, y_test)
        score = 0.5 * metrics["f1"] + 0.5 * metrics["roc_auc"]

        row = {"params": params, "metrics": metrics, "score": score}
        all_results.append(row)
        print(f"Candidate {params} => f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f} score={score:.4f}")

        if score > best_score:
            best_score = score
            best = (params, metrics, pipe)

    assert best is not None
    best_params, best_metrics, best_pipe = best

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipe, MODEL_PATH)
    joblib.dump(best_pipe, ALT_MODEL_PATH)

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_params": best_params,
        "best_metrics": best_metrics,
        "best_score": best_score,
        "all_results": all_results,
        "artifacts": {
            "runtime": str(MODEL_PATH),
            "backup": str(ALT_MODEL_PATH),
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"Saved runtime model: {MODEL_PATH}")
    print(f"Saved backup model:  {ALT_MODEL_PATH}")
    print(f"Saved report:        {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
