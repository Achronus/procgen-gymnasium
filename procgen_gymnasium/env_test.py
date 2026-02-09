import numpy as np
import pytest
from .env import ENV_NAMES, ProcgenEnv


@pytest.mark.parametrize("env_name", ["coinrun", "starpilot"])
def test_seeding(env_name):
    num_envs = 1

    def make_env(level_num):
        return ProcgenEnv(
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


@pytest.mark.parametrize("env_name", ["coinrun", "starpilot"])
def test_determinism(env_name):
    def collect_observations():
        rng = np.random.RandomState(0)
        env = ProcgenEnv(num_envs=2, env_name=env_name, rand_seed=23)
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


@pytest.mark.parametrize("env_name", ENV_NAMES)
@pytest.mark.parametrize("num_envs", [1, 2, 16])
def test_multi_speed(env_name, num_envs, benchmark):
    env = ProcgenEnv(num_envs=num_envs, env_name=env_name)

    actions = np.zeros(num_envs, dtype=np.int32)

    def rollout(max_steps):
        step_count = 0
        while step_count < max_steps:
            env.step(actions)
            step_count += 1

    benchmark(lambda: rollout(1000))
    env.close()
