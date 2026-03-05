from typing import Protocol
import pandas as pd


class SignalModel(Protocol):
    lookback: int

    def predict(self, df: pd.DataFrame, news: float):
        ...
