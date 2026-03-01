from dataclasses import dataclass, field
from collections import defaultdict
from config import MAX_DAILY_LOSS


@dataclass
class RiskEngine:
    max_daily_loss: float = MAX_DAILY_LOSS
    max_open_positions: int = 5
    max_symbol_positions: int = 1
    open_positions: int = 0
    symbol_positions: dict = field(default_factory=lambda: defaultdict(int))

    def can_open(self, symbol: str, daily_pnl: float) -> tuple[bool, str]:
        if daily_pnl <= -abs(self.max_daily_loss):
            return False, "daily_loss_limit"
        if self.open_positions >= self.max_open_positions:
            return False, "max_open_positions"
        if self.symbol_positions[symbol] >= self.max_symbol_positions:
            return False, "max_symbol_positions"
        return True, "ok"

    def on_open(self, symbol: str) -> None:
        self.open_positions += 1
        self.symbol_positions[symbol] += 1

    def on_close(self, symbol: str) -> None:
        if self.open_positions > 0:
            self.open_positions -= 1
        if self.symbol_positions[symbol] > 0:
            self.symbol_positions[symbol] -= 1
