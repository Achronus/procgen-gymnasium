"""Core environment tests: seeding, determinism, state save/load, rendering."""

import numpy as np
import pytest

from procgen_gym.env import ENV_NAMES, ProcgenVecEnv


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_seeding(env_name):
    num_envs = 1

    def make_env(level_num):
        return ProcgenVecEnv(
            num_envs=num_envs, env_name=env_name, num_levels=1, start_level=level_num
        )

    env1 = make_env(0)
    env2 = make_env(0)
    env3 = make_env(1)

    obs1, _, _, _, _ = env1.step(np.zeros(num_envs, dtype=np.int32))
    obs2, _, _, _, _ = env2.step(np.zeros(num_envs, dtype=np.int32))
    obs3, _, _, _, _ = env3.step(np.zeros(num_envs, dtype=np.int32))

    assert np.array_equal(obs1, obs2)
    assert not np.array_equal(obs1, obs3)

    env1.close()
    env2.close()
    env3.close()


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_determinism(env_name):
    def collect_observations():
        rng = np.random.RandomState(0)
        env = ProcgenVecEnv(num_envs=2, env_name=env_name, rand_seed=23)
        obs, _ = env.reset()
        obses = [obs]
        for _ in range(128):
            actions = rng.randint(low=0, high=15, size=(2,), dtype=np.int32)
            obs, rew, terminated, truncated, info = env.step(actions)
            obses.append(obs)
        env.close()
        return np.array(obses)

    obs1 = collect_observations()
    obs2 = collect_observations()
    assert np.array_equal(obs1, obs2)


def test_state_save_load(coinrun_vec):
    coinrun_vec.reset()

    for _ in range(10):
        coinrun_vec.step(np.array([0], dtype=np.int32))

    state = coinrun_vec.get_state()
    obs_saved, _, _, _, _ = coinrun_vec.step(np.array([0], dtype=np.int32))

    for _ in range(10):
        coinrun_vec.step(np.array([1], dtype=np.int32))

    coinrun_vec.set_state(state)
    obs_restored, _, _, _, _ = coinrun_vec.step(np.array([0], dtype=np.int32))
    assert np.array_equal(obs_saved, obs_restored)


def test_render_mode_rgb_array(coinrun_vec_rgb):
    coinrun_vec_rgb.reset()

    frames = coinrun_vec_rgb.render()
    assert frames is not None
    assert isinstance(frames, list)
    assert len(frames) == 1
    assert frames[0].ndim == 3  # (H, W, 3)
    assert frames[0].shape[-1] == 3


@pytest.mark.parametrize("env_name", ENV_NAMES)
@pytest.mark.parametrize("num_envs", [1, 2, 16])
def test_multi_speed(env_name, num_envs, benchmark):
    env = ProcgenVecEnv(num_envs=num_envs, env_name=env_name)

    actions = np.zeros(num_envs, dtype=np.int32)

    def rollout(max_steps):
        step_count = 0
        while step_count < max_steps:
            env.step(actions)
            step_count += 1

    benchmark(lambda: rollout(1000))
    env.close()
