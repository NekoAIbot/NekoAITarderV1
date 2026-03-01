#!/usr/bin/env python3
"""Basic feature-drift check using reference vs current windows."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.market_data import fetch_market_data
from app.models.feature_builder import build_features


def psi(expected, actual, bins=10):
    eps = 1e-6
    brk = np.quantile(expected, np.linspace(0, 1, bins + 1))
    brk[0] = -np.inf
    brk[-1] = np.inf
    e_hist, _ = np.histogram(expected, bins=brk)
    a_hist, _ = np.histogram(actual, bins=brk)
    e = e_hist / max(e_hist.sum(), 1)
    a = a_hist / max(a_hist.sum(), 1)
    return float(np.sum((a - e) * np.log((a + eps) / (e + eps))))


def run(symbol="EURUSD"):
    try:
        df = fetch_market_data(symbol, "1m", 1500)
    except Exception:
        idx = np.arange(1500)
        base = np.cumsum(np.random.randn(1500)) + 100
        import pandas as pd
        df = pd.DataFrame({"open": base, "high": base + 0.1, "low": base - 0.1, "close": base, "volume": np.random.randint(1, 100, 1500)}, index=idx)
    feats = build_features(df, 0.0)
    ref = feats.iloc[:700]
    cur = feats.iloc[-700:]
    alerts = {}
    for c in feats.columns:
        v = psi(ref[c].values, cur[c].values)
        if v > 0.2:
            alerts[c] = v
    print("Drift alerts:", alerts if alerts else "none")
    return alerts


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "EURUSD")
