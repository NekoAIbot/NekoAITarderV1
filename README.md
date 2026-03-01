# NekoAITarderV1

## Quick start

1. Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install vaderSentiment
```

2. Configure `.env` with Telegram/MT5/API values used by `config.py`.

3. Train production runtime model:

```bash
python scripts/train_production_model.py
```

This writes the live runtime model used by `app/models/ai_model.py` to:

- `app/models/models/xgb_model.joblib` (runtime artifact)
- `models/xgb_model_production.joblib` (backup copy)
- `models/xgb_model_training_report.json` (metrics/report)

4. Optional backtest smoke run:

```bash
python scripts/run_backtest.py
```

5. Start engine:

```bash
python run.py
```

The scheduler will run trading cycles and an automatic retrain/backtest cadence.


## Repo hygiene check

Run this before pushing:

```bash
python scripts/check_no_binaries.py
```
