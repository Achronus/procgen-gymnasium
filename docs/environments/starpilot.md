# StarPilot

![StarPilot](../../screenshots/starpilot.png)

|   |   |
|---|---|
| **Action Space** | `Discrete(15)` |
| **Observation Space** | `Box(0, 255, (64, 64, 3), uint8)` |
| **Import** | `gym.make("procgen_gym/procgen-starpilot-v0")` |

## Description

A simple side scrolling shooter game. Relatively challenging for humans to play since all enemies fire projectiles that directly target the player. An inability to dodge quickly leads to the player's demise. There are fast and slow enemies, stationary turrets with high health, clouds which obscure player vision, and impassable meteors.

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

The player receives rewards for destroying enemies.

## Episode End

The episode ends when the player is destroyed by enemy projectiles.

## Distribution Modes

Supports `"easy"` and `"hard"` modes.

## Usage

```python
import gymnasium as gym
import procgen_gymnasium

# Single environment
env = gym.make("procgen_gym/procgen-starpilot-v0")

# Vectorized
env = gym.make_vec("procgen_gym/procgen-starpilot-v0", num_envs=16)
```
