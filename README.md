# NekoAITarderV1

AI-powered trading bot for forex and crypto with model training, backtesting, and Telegram/MT5 integration.

## Production-grade training & backtesting scripts

This repository now includes hardened scripts for richer data usage and repeatable model lifecycle:

- `scripts/production_train.py`
  - Pulls data for all configured forex + crypto symbols.
  - Uses the existing multi-provider market-data fallback chain from `app.market_data`.
  - Builds robust technical and cross-market features.
  - Runs purged walk-forward validation across multiple model families.
  - Saves a self-contained model bundle to `models/production/best_production_bundle.joblib`.

- `scripts/production_backtest.py`
  - Loads the saved production bundle.
  - Recreates features from newly fetched data automatically.
  - Runs out-of-sample simulation with configurable spread/slippage/fees.
  - Outputs detailed JSON report to `models/production/latest_backtest.json`.

### Example usage

```bash
python scripts/production_train.py --timeframe 5m --limit 20000 --horizon 3
python scripts/production_backtest.py --bundle models/production/best_production_bundle.joblib --timeframe 5m --limit 20000
```

### Notes

- These scripts prioritize the richest available data by requesting deep history and relying on the built-in provider fallback sequence.
- If premium API keys (TwelveData/AlphaVantage) are configured, they are used automatically when higher-priority sources fail.
