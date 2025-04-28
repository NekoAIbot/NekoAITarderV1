# app/risk_manager.py

import json
from pathlib import Path
from config import LOT_BASE, LOT_MIN, LOT_MAX, LOT_ADJUST_PERCENT

STATE_FILE = Path(__file__).parent / "risk_state.json"

class RiskManager:
    """Tracks and adjusts your lot size, persisting between runs."""
    def __init__(self):
        self.file = STATE_FILE
        self.current_lot = LOT_BASE
        self._load_state()

    def _load_state(self):
        """Load last saved lot size, if available."""
        if self.file.exists():
            try:
                data = json.loads(self.file.read_text())
                self.current_lot = data.get("current_lot", LOT_BASE)
                print(f"ℹ️  RiskManager loaded lot size {self.current_lot:.4f}")
            except Exception:
                self.current_lot = LOT_BASE
        else:
            print(f"ℹ️  No existing risk state; starting at base lot {LOT_BASE:.4f}")

    def _save_state(self):
        """Save current lot size to disk."""
        try:
            self.file.write_text(json.dumps({"current_lot": self.current_lot}))
        except Exception:
            print("⚠️ Could not write risk state file.")

    def get_lot(self) -> float:
        """Return lot size for next trade."""
        return self.current_lot

    def adjust(self, last_trade_win: bool) -> float:
        """
        Adjust the lot up or down, then persist.
        - Win:  +ADJUST_PERCENT% (capped at LOT_MAX)
        - Loss: –ADJUST_PERCENT% (floored at LOT_MIN)
        """
        if last_trade_win:
            new = self.current_lot * (1 + LOT_ADJUST_PERCENT / 100)
        else:
            new = self.current_lot * (1 - LOT_ADJUST_PERCENT / 100)

        # clamp between your min/max
        self.current_lot = max(LOT_MIN, min(LOT_MAX, new))
        self._save_state()
        print(f"ℹ️  RiskManager adjusted lot to {self.current_lot:.4f}")
        return self.current_lot
