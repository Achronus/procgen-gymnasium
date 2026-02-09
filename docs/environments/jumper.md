# Jumper

![Jumper](../../screenshots/jumper.png)

|   |   |
|---|---|
| **Action Space** | `Discrete(15)` |
| **Observation Space** | `Box(0, 255, (64, 64, 3), uint8)` |
| **Import** | `gymnasium.make("procgen:procgen-jumper-v0")` |

## Description

A platformer with an open world layout. The player, a bunny, must navigate through the world to find the carrot. It might be necessary to ascend or descend the level to do so. The player is capable of "double jumping", allowing it to navigate tricky layouts and reach high platforms. There are spike obstacles which will destroy the player on contact. The screen includes a compass which displays direction and distance to the carrot.

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

The only reward comes from collecting the carrot, at which point the episode ends.

## Episode End

The episode ends when the player collects the carrot or is destroyed by a spike obstacle.

## Distribution Modes

Supports `"easy"` and `"hard"` modes. Also supports `"exploration"` mode.

## Known Issues

Due to a bug that permits the player to spawn on top of critical objects (an obstacle or the goal), ~7% of levels will terminate after a single action, the vast majority of which will have 0 reward.

## Usage

```python
import gymnasium
import procgen_gymnasium

# Single environment
env = gymnasium.make("procgen:procgen-jumper-v0")

# Vectorized
env = gymnasium.make_vec("procgen:procgen-jumper-v0", num_envs=16)
```
