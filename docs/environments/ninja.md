# Ninja

![Ninja](../../screenshots/ninja.png)

|   |   |
|---|---|
| **Action Space** | `Discrete(15)` |
| **Observation Space** | `Box(0, 255, (64, 64, 3), uint8)` |
| **Import** | `gym.make("procgen_gym/procgen-ninja-v0")` |

## Description

A simple platformer. The player, a ninja, must jump across narrow ledges while avoiding bomb obstacles. The player can toss throwing stars at several angles in order to clear bombs, if necessary. The player's jump can be charged over several timesteps to increase its effect.

## Action Space

The environment uses a `Discrete(15)` action space representing combinations of directional and button inputs:

| Action | Keys |
| --- | --- |
| 0 | LEFT + DOWN |
| 1 | LEFT |
| 2 | LEFT + UP |
| 3 | DOWN |
| 4 | (no-op) |
| 5 | UP |
| 6 | RIGHT + DOWN |
| 7 | RIGHT |
| 8 | RIGHT + UP |
| 9 | D |
| 10 | A |
| 11 | W |
| 12 | S |
| 13 | Q |
| 14 | E |

## Observation Space

The observation is an RGB image of shape `(64, 64, 3)` with pixel values in `[0, 255]` (`uint8`). By default, observations are centered on the agent.

## Rewards

The player receives a reward for collecting the mushroom at the end of the level.

## Episode End

The episode ends when the player collects the mushroom, falls off the level, or is hit by a bomb.

## Distribution Modes

Supports `"easy"` and `"hard"` modes. Also supports `"exploration"` mode.

## Usage

```python
import gymnasium as gym
import procgen_gymnasium

# Single environment
env = gym.make("procgen_gym/procgen-ninja-v0")

# Vectorized
env = gym.make_vec("procgen_gym/procgen-ninja-v0", num_envs=16)
```
