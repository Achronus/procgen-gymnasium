import os
import ctypes
import platform
import random
from typing import Sequence, Optional, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .builder import build
from .libenv import CLibenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_STATE_SIZE = 2 ** 20

ENV_NAMES = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]

EXPLORATION_LEVEL_SEEDS = {
    "coinrun": 1949448038,
    "caveflyer": 1259048185,
    "leaper": 1318677581,
    "jumper": 1434825276,
    "maze": 158988835,
    "heist": 876640971,
    "climber": 1561126160,
    "ninja": 1123500215,
}

# should match DistributionMode in game.h, except for 'exploration' which is handled by Python
DISTRIBUTION_MODE_DICT = {
    "easy": 0,
    "hard": 1,
    "extreme": 2,
    "memory": 10,
    "exploration": 20,
}


def create_random_seed():
    rand_seed = random.SystemRandom().randint(0, 2 ** 31 - 1)
    try:
        # force MPI processes to definitely choose different random seeds
        from mpi4py import MPI

        rand_seed = rand_seed - (rand_seed % MPI.COMM_WORLD.size) + MPI.COMM_WORLD.rank
    except ModuleNotFoundError:
        pass
    return rand_seed


KEY_COMBOS = [
    ("LEFT", "DOWN"),
    ("LEFT",),
    ("LEFT", "UP"),
    ("DOWN",),
    (),
    ("UP",),
    ("RIGHT", "DOWN"),
    ("RIGHT",),
    ("RIGHT", "UP"),
    ("D",),
    ("A",),
    ("W",),
    ("S",),
    ("Q",),
    ("E",),
]

_LIB_NAMES = ("libenv.so", "libenv.dylib", "env.dll")


def _lib_exists_in(directory):
    """Check if any recognized shared library file exists in the directory."""
    if not os.path.isdir(directory):
        return False
    return any(os.path.exists(os.path.join(directory, n)) for n in _LIB_NAMES)


def _find_lib_dir():
    """Locate the directory containing the prebuilt shared library.

    Search order:
      1. ``data/prebuilt/`` inside the installed package (pip wheel)
      2. Conda environment prefix lib directory
      3. ``None`` — triggers on-demand build fallback
    """
    # 1. pip / wheel install
    prebuilt = os.path.join(SCRIPT_DIR, "data", "prebuilt")
    if _lib_exists_in(prebuilt):
        return prebuilt

    # 2. conda-forge install
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        if platform.system() == "Windows":
            conda_lib = os.path.join(conda_prefix, "Library", "bin")
        else:
            conda_lib = os.path.join(conda_prefix, "lib")
        if _lib_exists_in(conda_lib):
            return conda_lib

    return None


class ProcgenVecEnv(gym.vector.VectorEnv):
    """
    Gymnasium VectorEnv wrapper around the procgen C++ library.

    This is a batched (vectorized) environment — it manages ``num_envs``
    sub-environments internally on the C++ side.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 15,
        "autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP,
    }

    def __init__(
        self,
        num_envs: int = 1,
        env_name: str = "coinrun",
        center_agent: bool = True,
        use_backgrounds: bool = True,
        use_monochrome_assets: bool = False,
        restrict_themes: bool = False,
        use_generated_assets: bool = False,
        paint_vel_info: bool = False,
        distribution_mode: str = "hard",
        debug: bool = False,
        rand_seed: Optional[int] = None,
        num_levels: int = 0,
        start_level: int = 0,
        use_sequential_levels: bool = False,
        debug_mode: int = 0,
        resource_root: Optional[str] = None,
        num_threads: int = 4,
        render_mode: Optional[str] = None,
    ):
        assert env_name in ENV_NAMES, f"Unknown environment: {env_name}"
        assert (
            distribution_mode in DISTRIBUTION_MODE_DICT
        ), f'"{distribution_mode}" is not a valid distribution mode.'

        if distribution_mode == "exploration":
            assert (
                env_name in EXPLORATION_LEVEL_SEEDS
            ), f"{env_name} does not support exploration mode"
            distribution_mode_int = DISTRIBUTION_MODE_DICT["hard"]
            num_levels = 1
            start_level = EXPLORATION_LEVEL_SEEDS[env_name]
        else:
            distribution_mode_int = DISTRIBUTION_MODE_DICT[distribution_mode]

        if resource_root is None:
            resource_root = os.path.join(SCRIPT_DIR, "data", "assets") + os.sep
            assert os.path.exists(resource_root), f"Asset root not found: {resource_root}"

        lib_dir = _find_lib_dir()
        if lib_dir is not None:
            assert not debug, "debug has no effect for pre-compiled library"
        else:
            lib_dir = build(debug=debug)

        if render_mode is None:
            render_human = False
        elif render_mode == "rgb_array":
            render_human = True
        else:
            raise ValueError(f"invalid render mode '{render_mode}', expected None or 'rgb_array'")

        if rand_seed is None:
            rand_seed = create_random_seed()

        options = {
            "env_name": env_name,
            "num_levels": num_levels,
            "start_level": start_level,
            "num_actions": len(KEY_COMBOS),
            "use_sequential_levels": use_sequential_levels,
            "debug_mode": debug_mode,
            "rand_seed": rand_seed,
            "num_threads": num_threads,
            "render_human": render_human,
            "resource_root": resource_root,
            "center_agent": center_agent,
            "use_generated_assets": use_generated_assets,
            "use_monochrome_assets": use_monochrome_assets,
            "restrict_themes": restrict_themes,
            "use_backgrounds": use_backgrounds,
            "paint_vel_info": paint_vel_info,
            "distribution_mode": distribution_mode_int,
        }

        self._clib = CLibenv(
            lib_dir=lib_dir,
            num=num_envs,
            options=options,
            c_func_defs=[
                "int get_state(libenv_env *, int, char *, int);",
                "void set_state(libenv_env *, int, char *, int);",
            ],
        )
        self._env_name = env_name

        # Initialize VectorEnv base (no-arg super().__init__ in gymnasium 1.x)
        super().__init__()

        # Set required VectorEnv attributes directly
        self.num_envs = num_envs
        self.render_mode = render_mode

        self.single_observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.single_action_space = spaces.Discrete(len(KEY_COMBOS))

        from gymnasium.vector.utils import batch_space
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """
        Observe the current state. Procgen environments auto-reset
        internally, so this returns the current observation.

        Returns:
            obs: np.ndarray of shape (num_envs, 64, 64, 3)
            info: dict of per-env info arrays
        """
        # Procgen auto-resets; we just observe the current state
        first, obs, _rew, info = self._clib.observe()
        return obs["rgb"], self._convert_info(info)

    def step(self, actions):
        """
        Step all environments with the given actions.

        Args:
            actions: np.ndarray of shape (num_envs,) with int actions

        Returns:
            obs, reward, terminated, truncated, info
        """
        self._clib.act(np.asarray(actions, dtype=np.int32))
        first, obs, rew, info = self._clib.observe()

        # In procgen, 'first' indicates the env was just reset (episode ended on the
        # *previous* step). We treat this as the episode having terminated.
        # The observation returned is already the first obs of the NEW episode (auto-reset).
        terminated = first.astype(bool)

        # Procgen doesn't distinguish truncation from termination
        truncated = np.zeros(self.num_envs, dtype=bool)

        return obs["rgb"], rew, terminated, truncated, self._convert_info(info)

    def _convert_info(self, raw_info):
        """Convert raw info dict to gymnasium-compatible info dict."""
        info = {}
        for key, arr in raw_info.items():
            info[key] = arr
        return info

    def render(self):
        """Return list of RGB frames (one per env) if render_mode='rgb_array'."""
        if self.render_mode == "rgb_array":
            info = self._clib.get_info_bufs()
            if "rgb" in info:
                frames = info["rgb"]
            else:
                frames = self._clib.get_ob_bufs()["rgb"]
            return [frames[i].copy() for i in range(self.num_envs)]
        return None

    def close(self):
        """Release C resources."""
        if hasattr(self, "_clib") and self._clib is not None:
            self._clib.close()
            self._clib = None

    # ---- State save/load (procgen-specific) ----

    def get_state(self):
        """Serialize the state of each sub-environment."""
        length = MAX_STATE_SIZE
        buf = ctypes.create_string_buffer(length)
        result = []
        for env_idx in range(self.num_envs):
            n = self._clib.call_c_func(
                "get_state", env_idx, buf, length
            )
            result.append(buf.raw[:n])
        return result

    def set_state(self, states):
        """Restore the state of each sub-environment."""
        assert len(states) == self.num_envs
        for env_idx in range(self.num_envs):
            state = states[env_idx]
            self._clib.call_c_func(
                "set_state", env_idx, state, len(state)
            )

    # ---- Interactive helper ----

    def keys_to_act(self, keys_list: Sequence[Sequence[str]]) -> List[Optional[np.ndarray]]:
        """Convert list of keys being pressed to actions, used in interactive mode."""
        result = []
        for keys in keys_list:
            action = None
            max_len = -1
            for i, combo in enumerate(KEY_COMBOS):
                pressed = all(key in keys for key in combo)
                if pressed and max_len < len(combo):
                    action = i
                    max_len = len(combo)
            if action is not None:
                action = np.array([action])
            result.append(action)
        return result


class ProcgenEnv(gym.Env):
    """
    Single-environment wrapper around ProcgenVecEnv.

    Provides standard ``gymnasium.Env`` interface (non-vectorized) so that
    ``gym.make("procgen_gym/procgen-coinrun-v0")`` works as expected.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, env_name: str = "coinrun", render_mode: Optional[str] = None, **kwargs):
        self._vec_env = ProcgenVecEnv(
            num_envs=1, env_name=env_name, num_threads=0,
            render_mode=render_mode, **kwargs,
        )
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(KEY_COMBOS))
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        obs, info = self._vec_env.reset(seed=seed, options=options)
        return obs[0], {k: v[0] for k, v in info.items()}

    def step(self, action):
        obs, rew, terminated, truncated, info = self._vec_env.step(np.array([action], dtype=np.int32))
        return obs[0], float(rew[0]), bool(terminated[0]), bool(truncated[0]), {k: v[0] for k, v in info.items()}

    def render(self):
        frame = self._vec_env.render()
        if frame is not None:
            return frame[0]
        return None

    def close(self):
        self._vec_env.close()
