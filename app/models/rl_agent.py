import os
from pathlib import Path
from stable_baselines3 import PPO
from app.models.rl_env import TradingEnv

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_FILE = MODEL_DIR / "rl_agent.zip"

class RLAgent:
    """
    A real PPO‐based trading agent.
    """
    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.model = PPO.load(str(MODEL_FILE)) if MODEL_FILE.exists() else None

    def train(self, df, total_timesteps=50_000):
        env = TradingEnv(df)
        self.model = PPO("MlpPolicy", env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(str(MODEL_FILE))

    def predict(self, df, news=None):
        env = TradingEnv(df)
        obs = env.reset()
        action, _ = self.model.predict(obs, deterministic=True)
        return {"signal": ["HOLD","BUY","SELL"][action], "confidence": 0.0, "predicted_change": 0.0}
