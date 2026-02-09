"""Tests for Gymnasium reward wrappers with ProcgenVecEnv."""

import numpy as np


def test_clip_reward(coinrun_vec2):
    from gymnasium.wrappers.vector import ClipReward

    env = ClipReward(coinrun_vec2, min_reward=-0.5, max_reward=0.5)
    env.reset()

    for _ in range(200):
        obs, rew, terminated, truncated, info = env.step(
            np.random.randint(0, 15, size=(2,), dtype=np.int32)
        )
        assert np.all(rew >= -0.5)
        assert np.all(rew <= 0.5)


def test_normalize_reward(coinrun_vec2):
    from gymnasium.wrappers.vector import NormalizeReward

    env = NormalizeReward(coinrun_vec2, gamma=0.99)
    env.reset()

    for _ in range(50):
        obs, rew, terminated, truncated, info = env.step(
            np.random.randint(0, 15, size=(2,), dtype=np.int32)
        )
    assert rew.shape == (2,)
    assert rew.dtype == np.float64


def test_transform_reward(coinrun_vec2):
    from gymnasium.wrappers.vector import TransformReward

    env = TransformReward(coinrun_vec2, func=lambda r: np.clip(r * 10.0, -1.0, 1.0))
    env.reset()

    for _ in range(50):
        obs, rew, terminated, truncated, info = env.step(
            np.random.randint(0, 15, size=(2,), dtype=np.int32)
        )
    assert rew.shape == (2,)
