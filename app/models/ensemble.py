# app/models/ensemble.py

from pathlib import Path
import joblib

# 1) Import your two pretrained pipelines
from app.models.xgb_model import MomentumModel as XGBModel
from app.models.ai_model  import MomentumModel as RFModel

# 2) Import RL trainer (install stable-baselines3 first)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 3) A tiny stub environment (you should replace this with your market‐simulation env)
class DummyEnv:
    def __init__(self):
        pass
    def reset(self):
        return [0.0]  # dummy obs
    def step(self, action):
        return [0.0], 0.0, True, {}  # obs, reward, done, info

# 4) Helper to build a vectorized env
def make_env():
    return DummyEnv()

class EnsembleModel:
    """
    Combines XGBoost, RF and an RL policy (PPO) into one ensemble.
    """
    MODEL_DIR    = Path(__file__).parent / "models"
    XGB_FILE     = MODEL_DIR / "xgb_model.joblib"
    RF_FILE      = MODEL_DIR / "rf_model.joblib"
    RL_FILE      = MODEL_DIR / "ppo_model.zip"

    def __init__(self):
        self.MODEL_DIR.mkdir(exist_ok=True, parents=True)

        # load or init XGB
        self.xgb = XGBModel()
        if self.XGB_FILE.exists():
            self.xgb.pipeline = joblib.load(self.XGB_FILE)

        # load or init RF
        self.rf = RFModel()
        if self.RF_FILE.exists():
            self.rf.pipeline = joblib.load(self.RF_FILE)

        # load or init PPO
        self.env = DummyVecEnv([make_env])
        if self.RL_FILE.exists():
            self.ppo = PPO.load(str(self.RL_FILE), env=self.env)
        else:
            self.ppo = None

    def predict(self, df, news):
        """
        Return a combined vote: XGB + RF + (PPO if available).
        """
        xgb_out = self.xgb.predict(df, news)
        rf_out  = self.rf.predict(df, news)

        votes = {"BUY": 0, "SELL": 0}
        votes[xgb_out["signal"]] += 1
        votes[rf_out["signal"]]  += 1

        # RL policy vote
        if self.ppo:
            obs = df.iloc[-1][["open","high","low","close","volume"]].values.tolist() + [news]
            act = self.ppo.predict(obs, deterministic=True)[0]
            rl_sig = "BUY" if act == 1 else "SELL"
            votes[rl_sig] += 1

        # majority vote
        final_sig = max(votes, key=votes.get)
        return {
            "signal": final_sig,
            "votes":  votes,
            "details": {"xgb": xgb_out, "rf": rf_out}
        }
