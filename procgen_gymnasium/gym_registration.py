import gymnasium
from .env import ENV_NAMES


def make_vec_env(env_name, render_mode=None, num_envs=1, **kwargs):
    """Factory function used by gymnasium.make_vec() to create a ProcgenVecEnv."""
    from .env import ProcgenVecEnv
    return ProcgenVecEnv(
        num_envs=num_envs, env_name=env_name, num_threads=0,
        render_mode=render_mode, **kwargs,
    )


def register_environments():
    for env_name in ENV_NAMES:
        gymnasium.register(
            id=f"procgen:procgen-{env_name}-v0",
            entry_point="procgen_gymnasium.env:ProcgenEnv",
            vector_entry_point="procgen_gymnasium.gym_registration:make_vec_env",
            kwargs={"env_name": env_name},
        )
