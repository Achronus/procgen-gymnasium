# BigFish

![BigFish](../../screenshots/bigfish.png)

|   |   |
|---|---|
| **Action Space** | `Discrete(15)` |
| **Observation Space** | `Box(0, 255, (64, 64, 3), uint8)` |
| **Import** | `gym.make("procgen_gym/procgen-bigfish-v0")` |

## Description

The player starts as a small fish and becomes bigger by eating other fish. The player may only eat fish smaller than itself, as determined solely by width. If the player comes in contact with a larger fish, the player is eaten and the episode ends.

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

The player receives a small reward for eating a smaller fish and a large reward for becoming bigger than all other fish.

## Episode End

The episode ends when the player is eaten by a larger fish, or when the player becomes bigger than all other fish.

## Distribution Modes

Supports `"easy"` and `"hard"` modes.

## Known Issues

It is possible for the player to occasionally become trapped along the borders of the environment.

## Usage

```python
import gymnasium as gym
import procgen_gymnasium

# Single environment
env = gym.make("procgen_gym/procgen-bigfish-v0")

# Vectorized
env = gym.make_vec("procgen_gym/procgen-bigfish-v0", num_envs=16)
```
