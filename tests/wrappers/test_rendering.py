"""Tests for Gymnasium rendering wrappers with ProcgenVecEnv."""

import os
import tempfile

import numpy as np
import pytest

from procgen_gymnasium.env import ProcgenVecEnv


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("env_name", ["coinrun"])
def test_record_video(env_name):
    pytest.importorskip("moviepy", reason="moviepy not installed")
    from gymnasium.wrappers.vector import RecordVideo

    with tempfile.TemporaryDirectory() as tmpdir:
        env = ProcgenVecEnv(
            num_envs=1, env_name=env_name, render_mode="rgb_array",
            num_levels=1, start_level=0,
        )
        env = RecordVideo(env, video_folder=tmpdir, episode_trigger=lambda ep: ep == 0)

        obs, info = env.reset()
        for _ in range(100):
            actions = np.random.randint(0, 15, size=(1,), dtype=np.int32)
            obs, rew, terminated, truncated, info = env.step(actions)

        env.close()

        video_files = [f for f in os.listdir(tmpdir) if f.endswith(".mp4")]
        assert len(video_files) > 0, f"No video files found in {tmpdir}"


def test_human_rendering(coinrun_vec_rgb):
    """Verify HumanRendering can be instantiated without error."""
    pytest.importorskip("pygame", reason="pygame not installed")
    pytest.importorskip("cv2", reason="opencv not installed")
    from gymnasium.wrappers.vector import HumanRendering

    if os.environ.get("DISPLAY") is None and os.name != "nt":
        pytest.skip("No display available")

    env = HumanRendering(coinrun_vec_rgb)
    env.close()
