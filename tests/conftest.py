import numpy as np
import pytest

from procgen_gymnasium.env import ENV_NAMES, ProcgenVecEnv


@pytest.fixture
def coinrun_vec():
    """Single-env ProcgenVecEnv on coinrun with deterministic seed."""
    env = ProcgenVecEnv(num_envs=1, env_name="coinrun", num_levels=1, start_level=0)
    yield env
    env.close()


@pytest.fixture
def coinrun_vec_rgb():
    """Single-env ProcgenVecEnv on coinrun with render_mode='rgb_array'."""
    env = ProcgenVecEnv(
        num_envs=1, env_name="coinrun", render_mode="rgb_array",
        num_levels=1, start_level=0,
    )
    yield env
    env.close()


@pytest.fixture
def coinrun_vec2():
    """Two-env ProcgenVecEnv on coinrun with deterministic seed."""
    env = ProcgenVecEnv(num_envs=2, env_name="coinrun", num_levels=1, start_level=0)
    yield env
    env.close()
