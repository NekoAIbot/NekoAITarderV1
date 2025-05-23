# app/risk_manager.py
import json
from pathlib import Path
from config import LOT_BASE, LOT_MIN, LOT_MAX, LOT_ADJUST_PERCENT

STATE_FILE = Path(__file__).parent / "risk_state.json"

class RiskManager:
    def __init__(self):
        self.file = STATE_FILE
        self.current_lot = LOT_BASE
        self._load_state()

    def _load_state(self):
        if self.file.exists():
            try:
                data = json.loads(self.file.read_text())
                self.current_lot = data.get("current_lot", LOT_BASE)
                print(f"ℹ️ Loaded lot {self.current_lot:.4f}")
            except:
                self.current_lot = LOT_BASE
        else:
            print(f"ℹ️ Start lot {LOT_BASE:.4f}")

    def _save_state(self):
        try:
            self.file.write_text(json.dumps({"current_lot":self.current_lot}))
        except:
            print("⚠️ Could not save risk state")

    def get_lot(self)->float:
        return self.current_lot

    def adjust(self, win: bool)->float:
        new = (self.current_lot * (1 + LOT_ADJUST_PERCENT/100)) if win \
              else (self.current_lot * (1 - LOT_ADJUST_PERCENT/100))
        self.current_lot = max(LOT_MIN, min(LOT_MAX, new))
        self._save_state()
        print(f"ℹ️ Adjusted lot {self.current_lot:.4f}")
        return self.current_lot
