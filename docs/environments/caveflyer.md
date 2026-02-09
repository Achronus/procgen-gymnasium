# CaveFlyer

![CaveFlyer](../../screenshots/caveflyer.png)

|   |   |
|---|---|
| **Action Space** | `Discrete(15)` |
| **Observation Space** | `Box(0, 255, (64, 64, 3), uint8)` |
| **Import** | `gymnasium.make("procgen:procgen-caveflyer-v0")` |

## Description

The player must navigate a network of caves to reach the exit. Player movement mimics the Atari game "Asteroids": the ship can rotate and travel forward or backward along the current axis. The majority of the reward comes from successfully reaching the end of the level, though additional reward can be collected by destroying target objects along the way with the ship's lasers. There are stationary and moving lethal obstacles throughout the level.

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

The player receives a small reward for destroying targets and a large reward for reaching the exit.

## Episode End

The episode ends when the player reaches the exit or collides with a lethal obstacle.

## Distribution Modes

Supports `"easy"` and `"hard"` modes. Also supports `"exploration"` mode.

## Known Issues

In ~0.5% of levels, the player spawns next to an enemy and will die in a single step regardless of which action is taken.

## Usage

```python
import gymnasium
import procgen_gymnasium

# Single environment
env = gymnasium.make("procgen:procgen-caveflyer-v0")

# Vectorized
env = gymnasium.make_vec("procgen:procgen-caveflyer-v0", num_envs=16)
```
