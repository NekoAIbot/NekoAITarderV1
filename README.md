# NekoAITarderV1

AI-powered trading bot for forex and crypto with model training, backtesting, and Telegram/MT5 integration.

## Data preparation (Parquet-first)

### 1) Download FX Dukascopy data to parquet

```bash
python scripts/download_fx_dukascopy_parquet.py \
  --pairs EURUSD USDJPY GBPUSD USDCHF AUDUSD \
  --start-year 2016 --end-year 2025 \
  --output-dir data/raw/fx
```

### 2) Download crypto OHLCV to parquet

```bash
python scripts/download_crypto_parquet.py \
  --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT ADAUSDT DOTUSDT XRPUSDT \
  --timeframe 1m --period 60d \
  --output-dir data/raw/crypto
```

## Production training & backtesting (using parquet data)

- `scripts/production_train.py`
  - Uses local parquet files first (`data/raw/fx`, `data/raw/crypto`).
  - In default mode (`--data-mode parquet-only`), training **fails fast** if parquet is missing/low quality.
  - Runs walk-forward CV, model selection, threshold search, and writes bundle metadata.

- `scripts/production_backtest.py`
  - Uses the same parquet-first data path.
  - Uses saved training threshold by default.
  - Runs cost-aware strict OOS simulation and writes JSON report.

### Example usage

```bash
python scripts/production_train.py --timeframe 1m --data-mode parquet-only --limit 10000
python scripts/production_backtest.py --bundle models/production/best_production_bundle.joblib --timeframe 1m --data-mode parquet-only
```
