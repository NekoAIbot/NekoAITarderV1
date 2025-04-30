# test_data.py
from app.market_data import fetch_market_data

for sym in ["EURUSD", "USDJPY", "USDCAD"]:
    df = fetch_market_data(sym)
    print(f"{sym}: {df.shape[0]} rows ({', '.join(df.columns)})")
