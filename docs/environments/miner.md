# Miner

![Miner](../../screenshots/miner.png)

|   |   |
|---|---|
| **Action Space** | `Discrete(15)` |
| **Observation Space** | `Box(0, 255, (64, 64, 3), uint8)` |
| **Import** | `gymnasium.make("procgen:procgen-miner-v0")` |

## Description

Inspired by the classic game "Boulder Dash". The player, a robot, can dig through dirt to move throughout the world. The world has gravity, and dirt supports boulders and diamonds. Boulders and diamonds will fall through free space and roll off each other. If a boulder or a diamond falls on the player, the game is over. The goal is to collect all the diamonds in the level and then proceed through the exit.

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

The player receives a small reward for collecting a diamond and a larger reward for completing the level.

## Episode End

The episode ends when the player reaches the exit after collecting all diamonds, or when crushed by a falling boulder or diamond.

## Distribution Modes

Supports `"easy"` and `"hard"` modes.

## Known Issues

There is a low probability of unsolvable level configurations, with either a diamond or the exit being unreachable.

## Usage

```python
import gymnasium
import procgen_gymnasium

# Single environment
env = gymnasium.make("procgen:procgen-miner-v0")

# Vectorized
env = gymnasium.make_vec("procgen:procgen-miner-v0", num_envs=16)
```
