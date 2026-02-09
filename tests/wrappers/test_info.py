"""Tests for Gymnasium info wrappers with ProcgenVecEnv."""

import numpy as np
import pytest

from procgen_gym.env import ENV_NAMES, ProcgenVecEnv


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_record_episode_statistics(env_name):
    from gymnasium.wrappers.vector import RecordEpisodeStatistics

    env = ProcgenVecEnv(num_envs=2, env_name=env_name, num_levels=1, start_level=0)
    env = RecordEpisodeStatistics(env)

    obs, info = env.reset()
    assert obs.shape == (2, 64, 64, 3)

    for _ in range(500):
        actions = np.random.randint(0, 15, size=(2,), dtype=np.int32)
        obs, rew, terminated, truncated, info = env.step(actions)

        if np.any(terminated):
            assert "episode" in info, (
                f"Expected 'episode' key in info, got: {list(info.keys())}"
            )
            assert "_episode" in info
            episode = info["episode"]
            assert "r" in episode  # episode return
            assert "l" in episode  # episode length
            assert "t" in episode  # episode time
            break

    env.close()


def test_dict_info_to_list(coinrun_vec2):
    from gymnasium.wrappers.vector import DictInfoToList, RecordEpisodeStatistics

    env = RecordEpisodeStatistics(coinrun_vec2)
    env = DictInfoToList(env)

    obs, info = env.reset()
    assert isinstance(info, list)
    assert len(info) == 2

    for _ in range(200):
        obs, rew, terminated, truncated, info = env.step(
            np.random.randint(0, 15, size=(2,), dtype=np.int32)
        )
        assert isinstance(info, list)
        assert len(info) == 2
