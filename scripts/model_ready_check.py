#!/usr/bin/env python3
"""Smoke check that runtime model artifact is loadable and inference-ready."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.ai_model import MomentumModel


def main():
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=200, freq="min")
    base = np.cumsum(np.random.randn(200)) + 100
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 0.1,
            "low": base - 0.1,
            "close": base,
            "volume": np.random.randint(1, 1000, 200),
        },
        index=idx,
    )

    model = MomentumModel()
    out = model.predict(df, news=0.0)
    print("Model ready:", out)


if __name__ == "__main__":
    main()
