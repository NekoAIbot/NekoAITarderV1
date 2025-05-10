import gym
from gym import spaces
import numpy as np

class TradingEnv(gym.Env):
    """
    Production-grade trading env:
      - Discrete actions: 0 = flat, 1 = long, 2 = short
      - Observation: lookback bars × [open, high, low, close, volume, position]
      - Reward: unrealized PnL change minus slippage/commission
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, lookback=20, commission=0.0002):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.lookback = lookback
        self.commission = commission
        self.position = 0  # -1 short, 0 flat, +1 long
        self.entry_price = None
        self.step_idx = lookback

        # actions: flat / long / short
        self.action_space = spaces.Discrete(3)
        # five price fields + current position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(lookback, 6), dtype=np.float32
        )

    def reset(self):
        self.position = 0
        self.entry_price = 0.0
        self.step_idx = self.lookback
        return self._obs()

    def _obs(self):
        window = self.df.iloc[self.step_idx-self.lookback:self.step_idx]
        arr = window[["open","high","low","close","volume"]].values
        pos_col = np.full((self.lookback,1), self.position, dtype=np.float32)
        return np.hstack([arr, pos_col])

    def step(self, action):
        """
        Apply discrete action:
          - if changing position, pay commission on notional
        Reward is PnL difference since last step.
        """
        price = float(self.df.at[self.step_idx, "close"])
        prev_position = self.position

        # map action → position
        self.position = {0:0, 1:1, 2:-1}[action]

        # if we opened/closed, pay commission
        reward = 0.0
        if self.position != prev_position:
            reward -= abs(price) * self.commission

        # step forward price movement
        next_price = float(self.df.at[self.step_idx+1, "close"])
        # unrealized PnL change
        reward += (next_price - price) * self.position

        self.step_idx += 1
        done = self.step_idx >= len(self.df)-1
        return (self._obs() if not done else None, reward, done, {})

    def render(self, mode="human"):
        print(f"Step {self.step_idx}: position={self.position}")

    def close(self):
        pass
