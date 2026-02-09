"""Tests for Gymnasium single-env wrappers with ProcgenEnv (gymnasium.make)."""

import gymnasium
import numpy as np
import pytest


@pytest.fixture
def coinrun_single():
    env = gymnasium.make("procgen:procgen-coinrun-v0")
    yield env
    env.close()


@pytest.fixture
def coinrun_single_rgb():
    env = gymnasium.make("procgen:procgen-coinrun-v0", render_mode="rgb_array")
    yield env
    env.close()


def test_frame_stack_observation(coinrun_single):
    from gymnasium.wrappers import FrameStackObservation

    env = FrameStackObservation(coinrun_single, stack_size=4)
    obs, info = env.reset()
    assert obs.shape == (4, 64, 64, 3)

    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (4, 64, 64, 3)


def test_delay_observation(coinrun_single):
    from gymnasium.wrappers import DelayObservation

    env = DelayObservation(coinrun_single, delay=2)
    obs, info = env.reset()
    assert obs.shape == (64, 64, 3)

    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (64, 64, 3)


def test_time_aware_observation(coinrun_single):
    from gymnasium.wrappers import TimeAwareObservation, TimeLimit

    # TimeAwareObservation requires the env to have a max_episode_steps
    env = TimeLimit(coinrun_single, max_episode_steps=100)
    env = TimeAwareObservation(env)
    obs, info = env.reset()
    assert obs is not None

    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs is not None


def test_time_limit(coinrun_single):
    from gymnasium.wrappers import TimeLimit

    env = TimeLimit(coinrun_single, max_episode_steps=10)
    obs, info = env.reset()

    for i in range(10):
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated:
            break

    # After 10 steps, should be truncated (if not terminated naturally)
    if not terminated:
        assert truncated


def test_max_and_skip_observation(coinrun_single):
    from gymnasium.wrappers import MaxAndSkipObservation

    env = MaxAndSkipObservation(coinrun_single, skip=4)
    obs, info = env.reset()
    assert obs.shape == (64, 64, 3)

    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (64, 64, 3)


def test_sticky_action(coinrun_single):
    from gymnasium.wrappers import StickyAction

    env = StickyAction(coinrun_single, repeat_action_probability=0.25)
    obs, info = env.reset()
    assert obs.shape == (64, 64, 3)

    for _ in range(20):
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (64, 64, 3)
        if terminated:
            break


def test_grayscale_observation_single(coinrun_single):
    from gymnasium.wrappers import GrayscaleObservation

    env = GrayscaleObservation(coinrun_single, keep_dim=True)
    obs, info = env.reset()
    assert obs.shape == (64, 64, 1)


def test_resize_observation_single(coinrun_single):
    from gymnasium.wrappers import ResizeObservation

    env = ResizeObservation(coinrun_single, shape=(32, 32))
    obs, info = env.reset()
    assert obs.shape == (32, 32, 3)


def test_record_episode_statistics_single(coinrun_single):
    from gymnasium.wrappers import RecordEpisodeStatistics

    env = RecordEpisodeStatistics(coinrun_single)
    obs, info = env.reset()

    for _ in range(500):
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            assert "episode" in info
            assert "r" in info["episode"]
            assert "l" in info["episode"]
            break


def test_render_collection(coinrun_single_rgb):
    from gymnasium.wrappers import RenderCollection

    env = RenderCollection(coinrun_single_rgb)
    obs, info = env.reset()

    for _ in range(5):
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated:
            break

    frames = env.render()
    assert isinstance(frames, list)
    assert len(frames) > 0
    # gymnasium.make wraps render output (may upscale), just check it's an RGB image
    assert frames[0].ndim == 3
    assert frames[0].shape[-1] == 3


def test_add_render_observation(coinrun_single_rgb):
    from gymnasium.wrappers import AddRenderObservation

    # render_only=False gives a dict with both original obs and rendered pixels
    env = AddRenderObservation(coinrun_single_rgb, render_only=False, render_key="pixels")
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert "pixels" in obs
    assert "state" in obs
    assert obs["state"].shape == (64, 64, 3)
    assert obs["pixels"].ndim == 3
    assert obs["pixels"].shape[-1] == 3


def test_add_render_observation_render_only(coinrun_single_rgb):
    from gymnasium.wrappers import AddRenderObservation

    # render_only=True (default) replaces obs with rendered frame
    env = AddRenderObservation(coinrun_single_rgb, render_only=True)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 3
    assert obs.shape[-1] == 3


# ---------------------------------------------------------------------------
# Multiple wrapper application tests
# ---------------------------------------------------------------------------

def test_same_wrapper_applied_twice(coinrun_single):
    """Verify the same wrapper type can be applied multiple times."""
    from gymnasium.wrappers import GrayscaleObservation, ResizeObservation

    env = ResizeObservation(coinrun_single, shape=(32, 32))
    env = ResizeObservation(env, shape=(16, 16))
    obs, info = env.reset()
    assert obs.shape == (16, 16, 3)

    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (16, 16, 3)


def test_multiple_different_wrappers_stacked():
    """Verify a realistic multi-wrapper stack on a single env."""
    from gymnasium.wrappers import (
        FrameStackObservation,
        GrayscaleObservation,
        RecordEpisodeStatistics,
        ResizeObservation,
        TimeLimit,
    )

    env = gymnasium.make("procgen:procgen-coinrun-v0")
    env = TimeLimit(env, max_episode_steps=50)
    env = RecordEpisodeStatistics(env)
    env = ResizeObservation(env, shape=(32, 32))
    env = GrayscaleObservation(env, keep_dim=True)
    env = FrameStackObservation(env, stack_size=4)

    obs, info = env.reset()
    assert obs.shape == (4, 32, 32, 1)

    for _ in range(50):
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (4, 32, 32, 1)
        if terminated or truncated:
            assert "episode" in info
            obs, info = env.reset()
            break

    env.close()
