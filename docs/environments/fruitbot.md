# FruitBot

![FruitBot](../../screenshots/fruitbot.png)

|   |   |
|---|---|
| **Action Space** | `Discrete(15)` |
| **Observation Space** | `Box(0, 255, (64, 64, 3), uint8)` |
| **Import** | `gym.make("procgen_gym/procgen-fruitbot-v0")` |

## Description

A scrolling game where the player controls a robot that must navigate between gaps in walls and collect fruit along the way. Half of the spawned objects are fruit (positive reward) and half are non-fruit (negative reward). Occasionally the player must use a key to unlock gates which block the way.

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

The player receives a positive reward for collecting fruit, and a larger negative reward for collecting non-fruit objects. A large reward is given for reaching the end of the level.

## Episode End

The episode ends when the player reaches the end of the level or is blocked.

## Distribution Modes

Supports `"easy"` and `"hard"` modes.

## Usage

```python
import gymnasium as gym
import procgen_gymnasium

# Single environment
env = gym.make("procgen_gym/procgen-fruitbot-v0")

# Vectorized
env = gym.make_vec("procgen_gym/procgen-fruitbot-v0", num_envs=16)
```
