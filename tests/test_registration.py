"""Tests for gym.make() and gym.make_vec() registration."""

import gymnasium as gym
import numpy as np
import pytest

from procgen_gym.env import ENV_NAMES


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_gymnasium_make_vec(env_name):
    env = gym.make_vec(f"procgen_gym/procgen-{env_name}-v0", num_envs=1)
    obs, info = env.reset()
    assert obs.shape == (1, 64, 64, 3)

    actions = np.array([env.action_space.sample()])
    obs, rew, terminated, truncated, info = env.step(actions)
    assert obs.shape == (1, 64, 64, 3)
    assert rew.shape == (1,)
    assert terminated.shape == (1,)
    assert truncated.shape == (1,)

    env.close()


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_gymnasium_make(env_name):
    env = gym.make(f"procgen_gym/procgen-{env_name}-v0")
    obs, info = env.reset()
    assert obs.shape == (64, 64, 3)

    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (64, 64, 3)
    assert isinstance(rew, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    env.close()
