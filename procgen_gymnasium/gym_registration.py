import gymnasium
from .env import ENV_NAMES, ProcgenEnv


def make_env(env_name, render_mode=None, **kwargs):
    """Factory function used by gymnasium.make() to create a single-env ProcgenEnv."""
    return ProcgenEnv(num_envs=1, env_name=env_name, num_threads=0, render_mode=render_mode, **kwargs)


def register_environments():
    for env_name in ENV_NAMES:
        gymnasium.register(
            id=f"procgen_gymnasium/{env_name}-v0",
            entry_point="procgen_gymnasium.gym_registration:make_env",
            kwargs={"env_name": env_name},
        )
