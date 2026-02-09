"""Tests for stacking multiple Gymnasium wrappers on ProcgenVecEnv."""

import numpy as np


def test_wrapper_stack(coinrun_vec2):
    """Verify a common wrapper stack works end-to-end."""
    from gymnasium.wrappers.vector import (
        ClipReward,
        DtypeObservation,
        GrayscaleObservation,
        RecordEpisodeStatistics,
        RescaleObservation,
    )

    env = RecordEpisodeStatistics(coinrun_vec2)
    env = GrayscaleObservation(env, keep_dim=True)
    env = DtypeObservation(env, dtype=np.float64)
    env = RescaleObservation(env, min_obs=0.0, max_obs=1.0)
    env = ClipReward(env, min_reward=-1.0, max_reward=1.0)

    obs, info = env.reset()
    assert obs.shape == (2, 64, 64, 1)
    assert obs.dtype == np.float64
    assert obs.max() <= 1.0

    for _ in range(200):
        obs, rew, terminated, truncated, info = env.step(
            np.random.randint(0, 15, size=(2,), dtype=np.int32)
        )
        assert obs.shape == (2, 64, 64, 1)
        assert np.all(rew >= -1.0) and np.all(rew <= 1.0)


def test_same_vector_wrapper_applied_twice():
    """Verify the same vector wrapper can be applied multiple times."""
    from gymnasium.wrappers.vector import ResizeObservation

    from procgen_gym.env import ProcgenVecEnv

    env = ProcgenVecEnv(num_envs=1, env_name="coinrun", num_levels=1, start_level=0)
    env = ResizeObservation(env, shape=(32, 32))
    env = ResizeObservation(env, shape=(16, 16))

    obs, info = env.reset()
    assert obs.shape == (1, 16, 16, 3)

    obs, rew, terminated, truncated, info = env.step(np.array([0], dtype=np.int32))
    assert obs.shape == (1, 16, 16, 3)

    env.close()


def test_mixed_obs_and_reward_wrappers():
    """Verify mixing observation and reward wrappers in different orders."""
    from gymnasium.wrappers.vector import (
        ClipReward,
        DtypeObservation,
        NormalizeReward,
        ResizeObservation,
    )

    from procgen_gym.env import ProcgenVecEnv

    env = ProcgenVecEnv(num_envs=2, env_name="coinrun", num_levels=1, start_level=0)
    env = ClipReward(env, min_reward=-1.0, max_reward=1.0)
    env = ResizeObservation(env, shape=(32, 32))
    env = NormalizeReward(env, gamma=0.99)
    env = DtypeObservation(env, dtype=np.float32)

    obs, info = env.reset()
    assert obs.shape == (2, 32, 32, 3)
    assert obs.dtype == np.float32

    for _ in range(20):
        obs, rew, terminated, truncated, info = env.step(
            np.random.randint(0, 15, size=(2,), dtype=np.int32)
        )
        assert obs.shape == (2, 32, 32, 3)
        assert obs.dtype == np.float32

    env.close()
