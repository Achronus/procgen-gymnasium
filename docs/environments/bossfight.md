# BossFight

![BossFight](../../screenshots/bossfight.png)

|   |   |
|---|---|
| **Action Space** | `Discrete(15)` |
| **Observation Space** | `Box(0, 255, (64, 64, 3), uint8)` |
| **Import** | `gym.make("procgen_gym/procgen-bossfight-v0")` |

## Description

The player controls a small starship and must destroy a much bigger boss starship. The boss randomly selects from a set of possible attacks when engaging the player. The player must dodge the incoming projectiles or be destroyed. The player can also use randomly scattered meteors for cover. After a set timeout, the boss becomes vulnerable and its shields go down. At this point, the player's projectile attacks will damage the boss. Once the boss receives a certain amount of damage, the player receives a reward, and the boss re-raises its shields.

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

The player receives a reward each time the boss's shields are broken and a large reward for destroying the boss entirely.

## Episode End

The episode ends when the player is destroyed or when the boss is fully defeated.

## Distribution Modes

Supports `"easy"` and `"hard"` modes.

## Usage

```python
import gymnasium as gym
import procgen_gym

# Single environment
env = gym.make("procgen_gym/procgen-bossfight-v0")

# Vectorized
env = gym.make_vec("procgen_gym/procgen-bossfight-v0", num_envs=16)
```
