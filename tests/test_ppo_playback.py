import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_ppo


class DummyEnv:
    def __init__(self, *args, **kwargs):
        self.render_calls = 0

    def reset(self):
        return np.zeros((3, 84, 84), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((3, 84, 84), dtype=np.uint8), 0.0, True, False, {}

    def render(self):
        self.render_calls += 1

    def close(self):
        return None


class DummyModel:
    def predict(self, obs, deterministic=True):
        return 0, None


def test_play_trained_agent_runs_without_crashing(monkeypatch, tmp_path):
    model_path = tmp_path / "dummy_model.zip"
    model_path.write_bytes(b"dummy")

    monkeypatch.setattr(train_ppo, "SkipFrame", lambda env, skip=4: env)
    monkeypatch.setattr(train_ppo, "DictToImageWrapper", lambda env: env)
    monkeypatch.setattr(train_ppo, "MarioEnv", lambda *args, **kwargs: DummyEnv())
    monkeypatch.setattr(train_ppo, "PPO", type("DummyPPO", (), {"load": staticmethod(lambda *args, **kwargs: DummyModel())}))

    train_ppo.play_trained_agent(
        model_path=str(model_path),
        level="Level1-1",
        num_episodes=1,
        max_steps=1,
        render=False,
        deterministic=True,
    )
