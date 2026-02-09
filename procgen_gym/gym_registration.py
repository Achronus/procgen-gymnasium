import gymnasium as gym

from .env import ENV_NAMES


def make_vec_env(env_name, render_mode=None, num_envs=1, **kwargs):
    """Factory function used by gym.make_vec() to create a ProcgenVecEnv."""
    from .env import ProcgenVecEnv

    return ProcgenVecEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_threads=0,
        render_mode=render_mode,
        **kwargs,
    )


def register_environments():
    for env_name in ENV_NAMES:
        gym.register(
            id=f"procgen_gym/procgen-{env_name}-v0",
            entry_point="procgen_gym.env:ProcgenEnv",
            vector_entry_point="procgen_gym.gym_registration:make_vec_env",
            kwargs={"env_name": env_name},
        )
