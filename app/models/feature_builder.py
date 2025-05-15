# app/models/feature_builder.py

import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame, news) -> pd.DataFrame:
    """
    Compute:
      ret1, ret3,
      ma5, ma20, ma_diff,
      rsi, atr, adx,
      bb_width, macd, macd_hist,
      obv, sentiment

    `df` must have columns: high, low, open, close, volume, and a DateTimeIndex
    If its index is not a DatetimeIndex, higher-timeframe features are skipped.
    `news` may be a scalar or a pd.Series aligned to df.index.
    """
    X = df.copy()

    # 1) returns
    X["ret1"]    = X["close"].pct_change(1)
    X["ret3"]    = X["close"].pct_change(3)

    # 2) moving averages
    X["ma5"]     = X["close"].rolling(5).mean()
    X["ma20"]    = X["close"].rolling(20).mean()
    X["ma_diff"] = X["ma5"] - X["ma20"]

    # 3) RSI(14)
    delta    = X["close"].diff()
    up       = delta.clip(lower=0)
    down     = -delta.clip(upper=0)
    rs       = up.rolling(14).mean() / down.rolling(14).mean()
    X["rsi"] = 100 - (100 / (1 + rs))

    # 4) ATR(14)
    tr = pd.concat([
        X["high"] - X["low"],
        (X["high"] - X["close"].shift()).abs(),
        (X["low"]  - X["close"].shift()).abs()
    ], axis=1).max(axis=1)
    X["atr"] = tr.rolling(14).mean()

    # 5) ADX(14)
    upm      = X["high"] - X["high"].shift()
    downm    = X["low"].shift() - X["low"]
    plus_dm  = np.where((upm > downm) & (upm > 0), upm, 0.0)
    minus_dm = np.where((downm > upm) & (downm > 0), downm, 0.0)
    plus_di  = 100 * pd.Series(plus_dm, index=X.index).rolling(14).mean() / X["atr"]
    minus_di = 100 * pd.Series(minus_dm, index=X.index).rolling(14).mean() / X["atr"]
    dx       = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0)
    X["adx"] = dx.rolling(14).mean()

    # 6) Bollinger Band width
    mb        = X["close"].rolling(20).mean()
    sd        = X["close"].rolling(20).std()
    X["bb_width"] = ((mb + 2 * sd) - (mb - 2 * sd)) / mb

    # 7) MACD
    ema12     = X["close"].ewm(span=12, adjust=False).mean()
    ema26     = X["close"].ewm(span=26, adjust=False).mean()
    macd      = ema12 - ema26
    sig_line  = macd.ewm(span=9, adjust=False).mean()
    X["macd"]      = macd
    X["macd_hist"] = macd - sig_line

    # 8) OBV
    X["obv"] = (np.sign(X["close"].diff()) * X["volume"].fillna(0)).cumsum()

    # 9) Higher‐timeframe example (only if DatetimeIndex)
    if isinstance(X.index, pd.DatetimeIndex):
        # e.g. 15-minute EMA of close
        X["ema15m"] = (
            X["close"]
            .resample("15T")
            .last()
            .ewm(span=15, adjust=False)
            .mean()
            .reindex(X.index, method="ffill")
        )
    # otherwise skip ema15m

    # 10) sentiment
    if isinstance(news, pd.Series):
        # align by position & fill missing
        news_aligned = news.reindex(X.index).fillna(0.0)
        X["sentiment"] = news_aligned.values
    else:
        X["sentiment"] = float(news)

    # drop any remaining NaNs
    feats = [
        "ret1","ret3","ma5","ma20","ma_diff",
        "rsi","atr","adx","bb_width",
        "macd","macd_hist","obv","sentiment"
    ]
    # include ema15m only if present
    if "ema15m" in X.columns:
        feats.append("ema15m")

    return X.dropna()[feats]
