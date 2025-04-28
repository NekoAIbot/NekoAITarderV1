# app/id_manager.py
import json
from pathlib import Path

ID_FILE = Path(__file__).parent / "id_state.json"

class IDManager:
    """Persistently hands out 1, 2, 3… signal IDs."""
    def __init__(self):
        self.file = ID_FILE
        self.current_id = 0
        self._load()

    def _load(self):
        if self.file.exists():
            try:
                payload = json.loads(self.file.read_text())
                self.current_id = payload.get("current_id", 0)
            except Exception:
                self.current_id = 0

    def next(self) -> int:
        self.current_id += 1
        try:
            self.file.write_text(json.dumps({"current_id": self.current_id}))
        except Exception:
            pass
        return self.current_id