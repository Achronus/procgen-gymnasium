# CoinRun

![CoinRun](../../screenshots/coinrun.png)

|   |   |
|---|---|
| **Action Space** | `Discrete(15)` |
| **Observation Space** | `Box(0, 255, (64, 64, 3), uint8)` |
| **Import** | `gymnasium.make("procgen:procgen-coinrun-v0")` |

## Description

A simple platformer. The goal is to collect the coin at the far right of the level, and the player spawns on the far left. The agent must dodge stationary saw obstacles, enemies that pace back and forth, and chasms that lead to death. Note that while the previously released version of CoinRun painted velocity information directly onto observations, the current version does not. This makes the environment significantly more difficult.

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

The player receives a reward for collecting the coin at the end of the level.

## Episode End

The episode ends when the player collects the coin, falls into a chasm, or is hit by an obstacle or enemy.

## Distribution Modes

Supports `"easy"` and `"hard"` modes. Also supports `"exploration"` mode.

## Usage

```python
import gymnasium
import procgen_gymnasium

# Single environment
env = gymnasium.make("procgen:procgen-coinrun-v0")

# Vectorized
env = gymnasium.make_vec("procgen:procgen-coinrun-v0", num_envs=16)
```
