# Chaser

![Chaser](../../screenshots/chaser.png)

|   |   |
|---|---|
| **Action Space** | `Discrete(15)` |
| **Observation Space** | `Box(0, 255, (64, 64, 3), uint8)` |
| **Import** | `gymnasium.make("procgen:procgen-chaser-v0")` |

## Description

Inspired by the Atari game "Ms. Pac-Man". Maze layouts are generated using Kruskal's algorithm, and then walls are removed until no dead-ends remain in the maze. The player must collect all the green orbs. Three large stars spawn that will make enemies vulnerable for a short time when collected. A collision with an enemy that isn't vulnerable results in the player's death. When a vulnerable enemy is eaten, an egg spawns somewhere on the map that will hatch into a new enemy after a short time, keeping the total number of enemies constant.

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

The player receives a small reward for collecting each orb and a large reward for completing the level.

## Episode End

The episode ends when the player collects all orbs, or when the player collides with a non-vulnerable enemy.

## Distribution Modes

Supports `"easy"` and `"hard"` modes.

## Usage

```python
import gymnasium
import procgen_gymnasium

# Single environment
env = gymnasium.make("procgen:procgen-chaser-v0")

# Vectorized
env = gymnasium.make_vec("procgen:procgen-chaser-v0", num_envs=16)
```
