"""Tests for Gymnasium observation wrappers with ProcgenVecEnv."""

import numpy as np

from procgen_gymnasium.env import ProcgenVecEnv


def test_grayscale_observation(coinrun_vec):
    from gymnasium.wrappers.vector import GrayscaleObservation

    env = GrayscaleObservation(coinrun_vec, keep_dim=True)
    obs, info = env.reset()
    assert obs.shape == (1, 64, 64, 1)
    assert obs.dtype == np.uint8

    obs, rew, terminated, truncated, info = env.step(np.array([0], dtype=np.int32))
    assert obs.shape == (1, 64, 64, 1)

    # Without keep_dim
    env2 = ProcgenVecEnv(num_envs=1, env_name="coinrun", num_levels=1, start_level=0)
    env2 = GrayscaleObservation(env2, keep_dim=False)
    obs2, _ = env2.reset()
    assert obs2.shape == (1, 64, 64)
    env2.close()


def test_resize_observation(coinrun_vec):
    from gymnasium.wrappers.vector import ResizeObservation

    env = ResizeObservation(coinrun_vec, shape=(32, 32))
    obs, info = env.reset()
    assert obs.shape == (1, 32, 32, 3)

    obs, rew, terminated, truncated, info = env.step(np.array([0], dtype=np.int32))
    assert obs.shape == (1, 32, 32, 3)


def test_reshape_observation(coinrun_vec):
    from gymnasium.wrappers.vector import ReshapeObservation

    env = ReshapeObservation(coinrun_vec, shape=(64, 192))
    obs, info = env.reset()
    assert obs.shape == (1, 64, 192)

    obs, rew, terminated, truncated, info = env.step(np.array([0], dtype=np.int32))
    assert obs.shape == (1, 64, 192)


def test_rescale_observation(coinrun_vec):
    from gymnasium.wrappers.vector import DtypeObservation, RescaleObservation

    # RescaleObservation needs float dtype to produce meaningful results
    env = DtypeObservation(coinrun_vec, dtype=np.float64)
    env = RescaleObservation(env, min_obs=0.0, max_obs=1.0)
    obs, info = env.reset()
    assert obs.shape == (1, 64, 64, 3)
    assert obs.dtype == np.float64
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0

    obs, rew, terminated, truncated, info = env.step(np.array([0], dtype=np.int32))
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


def test_dtype_observation(coinrun_vec):
    from gymnasium.wrappers.vector import DtypeObservation

    env = DtypeObservation(coinrun_vec, dtype=np.float32)
    obs, info = env.reset()
    assert obs.dtype == np.float32
    assert obs.shape == (1, 64, 64, 3)

    obs, rew, terminated, truncated, info = env.step(np.array([0], dtype=np.int32))
    assert obs.dtype == np.float32


def test_flatten_observation(coinrun_vec):
    from gymnasium.wrappers.vector import FlattenObservation

    env = FlattenObservation(coinrun_vec)
    obs, info = env.reset()
    assert obs.shape == (1, 64 * 64 * 3)

    obs, rew, terminated, truncated, info = env.step(np.array([0], dtype=np.int32))
    assert obs.shape == (1, 64 * 64 * 3)


def test_normalize_observation(coinrun_vec2):
    from gymnasium.wrappers.vector import NormalizeObservation

    env = NormalizeObservation(coinrun_vec2)
    obs, info = env.reset()
    assert obs.shape == (2, 64, 64, 3)
    assert obs.dtype == np.float64

    for _ in range(10):
        obs, rew, terminated, truncated, info = env.step(
            np.random.randint(0, 15, size=(2,), dtype=np.int32)
        )
    assert obs.shape == (2, 64, 64, 3)


def test_transform_observation(coinrun_vec):
    from gymnasium.wrappers.vector import TransformObservation

    env = TransformObservation(coinrun_vec, func=lambda obs: obs.astype(np.float32) / 255.0)
    obs, info = env.reset()
    assert obs.dtype == np.float32
    assert obs.max() <= 1.0
