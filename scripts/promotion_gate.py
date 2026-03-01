#!/usr/bin/env python3
"""Promotion gate: combines walk-forward and drift checks for a simple pass/fail."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.walk_forward_cv import walk_forward
from scripts.drift_check import run as drift_run

MIN_WF_WINRATE = 0.52
MAX_DRIFT_ALERTS = 3


def main():
    wf = walk_forward("EURUSD")
    drift_alerts = drift_run("EURUSD")
    ok = wf >= MIN_WF_WINRATE and len(drift_alerts) <= MAX_DRIFT_ALERTS
    print(f"Promotion gate => {'PASS' if ok else 'FAIL'} | wf={wf:.4f} alerts={len(drift_alerts)}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
