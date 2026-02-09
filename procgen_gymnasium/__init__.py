from importlib.metadata import version

__version__ = version("procgen-gymnasium")

from .env import ProcgenEnv, ProcgenVecEnv
from .gym_registration import register_environments

register_environments()

__all__ = ["ProcgenEnv", "ProcgenVecEnv"]
