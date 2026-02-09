# Procgen Gymnasium

A [Gymnasium](https://gymnasium.farama.org/) port of [OpenAI's Procgen Benchmark](https://github.com/openai/procgen) — 16 procedurally-generated environments for Reinforcement Learning (RL) research.

[[Original Paper]](https://arxiv.org/abs/1912.01588) [[Original Blog Post]](https://openai.com/blog/procgen-benchmark/)

![Procgen Environments](screenshots/procgen.gif)

## What is this?

This project modernises the original [procgen](https://github.com/openai/procgen) package (which depends on the deprecated `gym` and unmaintained `gym3` libraries) to work with **[Gymnasium](https://gymnasium.farama.org/)** (the maintained successor to OpenAI Gym) and **Python 3.13+**.

The C++ game code is **unchanged** — only the Python interface layer has been rewritten.

### Key changes from the original

- `gym` / `gym3` replaced with `gymnasium`
- Supports `gymnasium.make()` (single env) and `gymnasium.make_vec()` (vectorized)
- Compatible with all standard Gymnasium wrappers
- Python 3.13+ support
- Managed with `uv` and `pyproject.toml`

## Installation

```bash
pip install procgen-gym
```

**Requirements:** Python 3.13+ (64-bit)

## Quick Start

### Single environment (standard Gymnasium API)

```python
import gymnasium as gym
import procgen_gymnasium  # registers environments

env = gym.make("procgen_gym/procgen-coinrun-v0")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Vectorized environment (native batched)

```python
import gymnasium as gym
import numpy as np
import procgen_gymnasium  # registers environments

env = gym.make_vec("procgen_gym/procgen-coinrun-v0", num_envs=16)
obs, info = env.reset()

for _ in range(1000):
    actions = np.array([env.action_space.sample() for _ in range(16)])
    obs, reward, terminated, truncated, info = env.step(actions)

env.close()
```

### Direct instantiation

```python
from procgen_gymnasium import ProcgenEnv, ProcgenVecEnv

# Single environment
env = ProcgenEnv(env_name="coinrun")

# Vectorized environment (C++ level batching)
vec_env = ProcgenVecEnv(num_envs=16, env_name="coinrun")
```

## Environments

All environments produce `(64, 64, 3)` RGB observations and use a `Discrete(15)` action space. See [docs/environments/](docs/environments/) for detailed per-environment documentation.

| Screenshot | Name | Description |
| --- | --- | --- |
| <img src="screenshots/bigfish.png" width="120" alt="BigFish"> | [`bigfish`](docs/environments/bigfish.md) | Eat smaller fish to grow, avoid larger fish |
| <img src="screenshots/bossfight.png" width="120" alt="BossFight"> | [`bossfight`](docs/environments/bossfight.md) | Dodge projectiles and destroy a boss starship |
| <img src="screenshots/caveflyer.png" width="120" alt="CaveFlyer"> | [`caveflyer`](docs/environments/caveflyer.md) | Navigate cave networks in an Asteroids-style ship |
| <img src="screenshots/chaser.png" width="120" alt="Chaser"> | [`chaser`](docs/environments/chaser.md) | Collect orbs in a maze while avoiding enemies (Pac-Man inspired) |
| <img src="screenshots/climber.png" width="120" alt="Climber"> | [`climber`](docs/environments/climber.md) | Climb platforms and collect stars while dodging enemies |
| <img src="screenshots/coinrun.png" width="120" alt="CoinRun"> | [`coinrun`](docs/environments/coinrun.md) | Platformer — reach the coin at the far right |
| <img src="screenshots/dodgeball.png" width="120" alt="DodgeBall"> | [`dodgeball`](docs/environments/dodgeball.md) | Navigate rooms and throw balls at enemies (Berzerk inspired) |
| <img src="screenshots/fruitbot.png" width="120" alt="FruitBot"> | [`fruitbot`](docs/environments/fruitbot.md) | Collect fruit, avoid non-fruit objects in a scrolling game |
| <img src="screenshots/heist.png" width="120" alt="Heist"> | [`heist`](docs/environments/heist.md) | Collect coloured keys to unlock doors and steal the gem |
| <img src="screenshots/jumper.png" width="120" alt="Jumper"> | [`jumper`](docs/environments/jumper.md) | Open-world platformer with double-jump to find the carrot |
| <img src="screenshots/leaper.png" width="120" alt="Leaper"> | [`leaper`](docs/environments/leaper.md) | Cross lanes of traffic and hop across river logs (Frogger inspired) |
| <img src="screenshots/maze.png" width="120" alt="Maze"> | [`maze`](docs/environments/maze.md) | Navigate a procedurally-generated maze to find cheese |
| <img src="screenshots/miner.png" width="120" alt="Miner"> | [`miner`](docs/environments/miner.md) | Dig through dirt, avoid falling boulders, collect diamonds (Boulder Dash inspired) |
| <img src="screenshots/ninja.png" width="120" alt="Ninja"> | [`ninja`](docs/environments/ninja.md) | Precision platformer with charged jumps and throwing stars |
| <img src="screenshots/plunder.png" width="120" alt="Plunder"> | [`plunder`](docs/environments/plunder.md) | Naval cannon combat — destroy enemies, spare friendlies |
| <img src="screenshots/starpilot.png" width="120" alt="StarPilot"> | [`starpilot`](docs/environments/starpilot.md) | Side-scrolling space shooter with targeting enemies |

## Environment Options

All options can be passed as keyword arguments to `gym.make()`, `gym.make_vec()`, or the `ProcgenVecEnv` constructor:

| Option | Default | Description |
| --- | --- | --- |
| `num_levels` | `0` | Number of unique levels (0 = unlimited) |
| `start_level` | `0` | Lowest seed used to generate levels |
| `distribution_mode` | `"hard"` | Level variant: `"easy"`, `"hard"`, `"extreme"`, `"memory"`, `"exploration"` |
| `paint_vel_info` | `False` | Paint player velocity info in top left corner |
| `use_generated_assets` | `False` | Use randomly generated assets instead of designed ones |
| `use_backgrounds` | `True` | Use designed backgrounds (`False` for pure black) |
| `restrict_themes` | `False` | Restrict to single asset theme |
| `use_monochrome_assets` | `False` | Use monochromatic rectangles instead of designed assets |
| `center_agent` | `True` | Center observations on the agent |
| `use_sequential_levels` | `False` | Don't end episode at level completion, continue to next |
| `num_threads` | `4` | Number of C++ threads for environment stepping |
| `rand_seed` | Random | Random seed for level generation |
| `debug_mode` | `0` | Debug flag passed through to C++ game code |

## Gymnasium Wrapper Compatibility

`ProcgenVecEnv` and `ProcgenEnv` are compatible with standard Gymnasium wrappers.

### Vector wrappers (`gymnasium.wrappers.vector`)

| Wrapper | Compatible | Notes |
| --- | --- | --- |
| `GrayscaleObservation` | Yes | RGB to grayscale conversion |
| `ResizeObservation` | Yes | Resize 64x64 to any resolution |
| `ReshapeObservation` | Yes | Reshape observation arrays |
| `RescaleObservation` | Yes | Best used after `DtypeObservation(float)` |
| `DtypeObservation` | Yes | Convert uint8 to float32/float64 |
| `FlattenObservation` | Yes | Flatten to 1D (12288,) |
| `NormalizeObservation` | Yes | Running mean/std normalisation |
| `TransformObservation` | Yes | Custom observation function |
| `ClipReward` | Yes | Bound rewards to a range |
| `NormalizeReward` | Yes | Normalise rewards via running stats |
| `TransformReward` | Yes | Custom reward function |
| `TransformAction` | Yes | Custom action function |
| `RecordEpisodeStatistics` | Yes | Track episode returns and lengths |
| `RecordVideo` | Yes | Requires `render_mode="rgb_array"` and `moviepy` |
| `HumanRendering` | Yes | Requires `render_mode="rgb_array"`, `pygame`, and `opencv` |
| `DictInfoToList` | Yes | Convert info dict to list of dicts |
| `ClipAction` | No | Discrete action space (not Box) |
| `RescaleAction` | No | Discrete action space (not Box) |
| `FilterObservation` | No | Box observation space (not Dict) |

### Single-env wrappers (`gymnasium.wrappers`)

| Wrapper | Compatible | Notes |
| --- | --- | --- |
| `FrameStackObservation` | Yes | Stack N recent frames |
| `DelayObservation` | Yes | Add observation delay |
| `TimeAwareObservation` | Yes | Append time step to observation |
| `TimeLimit` | Yes | Truncate after N steps |
| `MaxAndSkipObservation` | Yes | Frame skipping with max pooling |
| `StickyAction` | Yes | Probabilistic action repeat |
| `GrayscaleObservation` | Yes | RGB to grayscale |
| `ResizeObservation` | Yes | Resize observations |
| `RecordEpisodeStatistics` | Yes | Episode tracking |
| `RenderCollection` | Yes | Collect rendered frames |
| `AddRenderObservation` | Yes | Add rendered frame to obs dict |

## State Save/Load

```python
from procgen_gymnasium import ProcgenVecEnv

env = ProcgenVecEnv(num_envs=1, env_name="coinrun")
env.reset()

# Save state
states = env.get_state()

# ... take some actions ...

# Restore state
env.set_state(states)
```

## Interactive Play

```bash
procgen-interactive --env-name coinrun
```

Keys: arrow keys + Q, W, E, A, S, D for actions. Score is displayed on screen.

## Building from Source

```bash
git clone https://github.com/Achronus/procgen-gymnasium.git
cd procgen-gymnasium
uv sync --extra test
uv run python -c "from procgen_gymnasium import ProcgenVecEnv; env = ProcgenVecEnv(num_envs=1, env_name='coinrun'); print('OK'); env.close()"
```

The C++ code requires [Qt5](https://www.qt.io/) for rendering. On Windows, install via [vcpkg](https://vcpkg.io/):

```bash
vcpkg install qt5-base:x64-windows
set PROCGEN_CMAKE_PREFIX_PATH=path/to/vcpkg_installed/x64-windows/share/cmake
```

## Known Issues

These are inherited from the original procgen and kept for reproducibility:

- **bigfish** — Player can occasionally become trapped along environment borders
- **caveflyer** — ~0.5% of levels spawn the player next to an enemy (instant death)
- **jumper** — ~7% of levels spawn the player on top of an enemy or goal (instant termination)
- **miner** — Low probability of unsolvable configurations

## Citation

```bibtex
@article{cobbe2019procgen,
  title={Leveraging Procedural Generation to Benchmark Reinforcement Learning},
  author={Cobbe, Karl and Hesse, Christopher and Hilton, Jacob and Schulman, John},
  journal={arXiv preprint arXiv:1912.01588},
  year={2019}
}
```
