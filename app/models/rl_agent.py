import gym
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_FILE = MODEL_DIR / "rl_agent.zip"

class RLAgent:
    def __init__(self, lookback=20):
        self.lookback = lookback
        MODEL_DIR.mkdir(exist_ok=True)
        if MODEL_FILE.exists():
            self.agent = PPO.load(str(MODEL_FILE))
        else:
            # build a toy gym.Env around your market data
            from .rl_env import TradingEnv
            env = DummyVecEnv([lambda: TradingEnv()])
            self.agent = PPO("MlpPolicy", env, verbose=0)

    def fit(self, df: pd.DataFrame, news: pd.Series):
        from .rl_env import TradingEnv
        env = DummyVecEnv([lambda: TradingEnv(df=df, news=news)])
        self.agent.set_env(env)
        self.agent.learn(total_timesteps=10_000)
        self.agent.save(str(MODEL_FILE))

    def predict(self, df: pd.DataFrame, news: float):
        from .rl_env import TradingEnv
        env = DummyVecEnv([lambda: TradingEnv(df=df, news=news)])
        action, _ = self.agent.predict(env.reset())
        return {"signal": "BUY" if action==1 else "SELL", "confidence": 100.0}
