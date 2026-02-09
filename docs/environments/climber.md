# Climber

![Climber](../../screenshots/climber.png)

|   |   |
|---|---|
| **Action Space** | `Discrete(15)` |
| **Observation Space** | `Box(0, 255, (64, 64, 3), uint8)` |
| **Import** | `gymnasium.make("procgen:procgen-climber-v0")` |

## Description

A simple platformer. The player must climb a sequence of platforms, collecting stars along the way. There are lethal flying monsters scattered throughout the level.

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

A small reward is given for collecting a star, and a larger reward is given for collecting all stars in a level.

## Episode End

If all stars are collected, the episode ends. The episode also ends if the player is hit by a flying monster.

## Distribution Modes

Supports `"easy"` and `"hard"` modes. Also supports `"exploration"` mode.

## Usage

```python
import gymnasium
import procgen_gymnasium

# Single environment
env = gymnasium.make("procgen:procgen-climber-v0")

# Vectorized
env = gymnasium.make_vec("procgen:procgen-climber-v0", num_envs=16)
```
