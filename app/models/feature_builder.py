# app/models/feature_builder.py

import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame, news) -> pd.DataFrame:
    """
    Compute: ret1, ret3, ma5, ma20, ma_diff,
             rsi, atr, adx, bb_width, macd, macd_hist, obv, sentiment
    `news` may be a scalar or a pd.Series aligned to df.index.
    """
    X = df.copy().reset_index(drop=True)
    # returns
    X["ret1"]    = X["close"].pct_change(1)
    X["ret3"]    = X["close"].pct_change(3)
    # moving avgs
    X["ma5"]     = X["close"].rolling(5).mean()
    X["ma20"]    = X["close"].rolling(20).mean()
    X["ma_diff"] = X["ma5"] - X["ma20"]
    # RSI(14)
    delta    = X["close"].diff()
    up       = delta.clip(lower=0)
    down     = -delta.clip(upper=0)
    rs       = up.rolling(14).mean() / down.rolling(14).mean()
    X["rsi"] = 100 - (100 / (1 + rs))
    # ATR(14)
    tr = pd.concat([
        X["high"] - X["low"],
        (X["high"] - X["close"].shift()).abs(),
        (X["low"]  - X["close"].shift()).abs()
    ], axis=1).max(axis=1)
    X["atr"] = tr.rolling(14).mean()
    # ADX(14)
    upm      = X["high"] - X["high"].shift()
    downm    = X["low"].shift() - X["low"]
    plus_dm  = np.where((upm>downm)&(upm>0), upm, 0.0)
    minus_dm = np.where((downm>upm)&(downm>0), downm, 0.0)
    plus_di  = 100 * pd.Series(plus_dm).rolling(14).mean() / X["atr"]
    minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / X["atr"]
    dx       = 100 * (abs(plus_di-minus_di)/(plus_di+minus_di)).replace([np.inf,-np.inf],0)
    X["adx"] = dx.rolling(14).mean()
    # Bollinger width
    mb        = X["close"].rolling(20).mean()
    sd        = X["close"].rolling(20).std()
    X["bb_width"] = ((mb + 2*sd) - (mb - 2*sd)) / mb
    # MACD
    ema12     = X["close"].ewm(span=12, adjust=False).mean()
    ema26     = X["close"].ewm(span=26, adjust=False).mean()
    macd      = ema12 - ema26
    sig_line  = macd.ewm(span=9, adjust=False).mean()
    X["macd"]      = macd
    X["macd_hist"] = macd - sig_line
    # OBV
    X["obv"] = (np.sign(X["close"].diff()) * X["volume"].fillna(0)).cumsum()
    # sentiment
    if isinstance(news, pd.Series):
        X["sentiment"] = news.reset_index(drop=True).reindex(X.index).fillna(0.0).values
    else:
        X["sentiment"] = float(news)
    # drop NaNs and select
    feats = [
        "ret1","ret3","ma5","ma20","ma_diff",
        "rsi","atr","adx","bb_width",
        "macd","macd_hist","obv","sentiment"
    ]
    return X.dropna()[feats]
