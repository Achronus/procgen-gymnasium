"""Tests for Gymnasium action wrappers with ProcgenVecEnv."""

import numpy as np


def test_transform_action(coinrun_vec):
    from gymnasium.wrappers.vector import TransformAction

    # Identity transform — just verify it applies without error
    env = TransformAction(coinrun_vec, func=lambda a: a)
    obs, info = env.reset()
    assert obs.shape == (1, 64, 64, 3)

    obs, rew, terminated, truncated, info = env.step(np.array([0], dtype=np.int32))
    assert obs.shape == (1, 64, 64, 3)
