# DodgeBall

![DodgeBall](../../screenshots/dodgeball.png)

|   |   |
|---|---|
| **Action Space** | `Discrete(15)` |
| **Observation Space** | `Box(0, 255, (64, 64, 3), uint8)` |
| **Import** | `gym.make("procgen_gym/procgen-dodgeball-v0")` |

## Description

Loosely inspired by the Atari game "Berzerk". The player spawns in a room with a random configuration of walls and enemies. Touching a wall loses the game and ends the episode. The player moves relatively slowly and can navigate throughout the room. There are enemies which also move slowly and which will occasionally throw balls at the player. The player can also throw balls, but only in the direction they are facing. If all enemies are hit, the player can move to the unlocked platform and earn a significant level completion bonus.

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

The player receives a reward for hitting enemies with balls and a large reward for clearing all enemies and reaching the platform.

## Episode End

The episode ends when the player touches a wall, is hit by an enemy ball, or clears the level.

## Distribution Modes

Supports `"easy"` and `"hard"` modes.

## Usage

```python
import gymnasium as gym
import procgen_gym

# Single environment
env = gym.make("procgen_gym/procgen-dodgeball-v0")

# Vectorized
env = gym.make_vec("procgen_gym/procgen-dodgeball-v0", num_envs=16)
```
