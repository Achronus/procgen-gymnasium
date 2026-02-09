# Procgen Environments

All 16 environments share the same interface:

- **Observation Space:** `Box(0, 255, (64, 64, 3), uint8)` — 64x64 RGB image
- **Action Space:** `Discrete(15)` — directional + button combinations
- **Render FPS:** 15 Hz (expected human play rate)

Levels are procedurally generated, providing a direct measure of how quickly a reinforcement learning agent learns generalisable skills.

## Environment List

| | Environment | Inspired By | Description |
| --- | --- | --- | --- |
| <img src="../../screenshots/bigfish.png" width="64" alt="BigFish"> | [BigFish](bigfish.md) | — | Eat smaller fish to grow, avoid larger fish |
| <img src="../../screenshots/bossfight.png" width="64" alt="BossFight"> | [BossFight](bossfight.md) | — | Dodge projectiles and destroy a boss starship |
| <img src="../../screenshots/caveflyer.png" width="64" alt="CaveFlyer"> | [CaveFlyer](caveflyer.md) | Asteroids | Navigate cave networks to reach the exit |
| <img src="../../screenshots/chaser.png" width="64" alt="Chaser"> | [Chaser](chaser.md) | Ms. Pac-Man | Collect orbs in a maze while avoiding enemies |
| <img src="../../screenshots/climber.png" width="64" alt="Climber"> | [Climber](climber.md) | — | Climb platforms and collect stars |
| <img src="../../screenshots/coinrun.png" width="64" alt="CoinRun"> | [CoinRun](coinrun.md) | — | Platformer — reach the coin at the far right |
| <img src="../../screenshots/dodgeball.png" width="64" alt="DodgeBall"> | [DodgeBall](dodgeball.md) | Berzerk | Navigate rooms and throw balls at enemies |
| <img src="../../screenshots/fruitbot.png" width="64" alt="FruitBot"> | [FruitBot](fruitbot.md) | — | Collect fruit, avoid non-fruit objects |
| <img src="../../screenshots/heist.png" width="64" alt="Heist"> | [Heist](heist.md) | — | Collect keys, unlock doors, steal the gem |
| <img src="../../screenshots/jumper.png" width="64" alt="Jumper"> | [Jumper](jumper.md) | — | Open-world platformer with double-jump |
| <img src="../../screenshots/leaper.png" width="64" alt="Leaper"> | [Leaper](leaper.md) | Frogger | Cross traffic lanes and hop across river logs |
| <img src="../../screenshots/maze.png" width="64" alt="Maze"> | [Maze](maze.md) | — | Navigate a maze to find cheese |
| <img src="../../screenshots/miner.png" width="64" alt="Miner"> | [Miner](miner.md) | Boulder Dash | Dig through dirt, collect diamonds, avoid boulders |
| <img src="../../screenshots/ninja.png" width="64" alt="Ninja"> | [Ninja](ninja.md) | — | Precision platformer with charged jumps |
| <img src="../../screenshots/plunder.png" width="64" alt="Plunder"> | [Plunder](plunder.md) | — | Naval cannon combat — target enemies, spare friendlies |
| <img src="../../screenshots/starpilot.png" width="64" alt="StarPilot"> | [StarPilot](starpilot.md) | — | Side-scrolling space shooter |

## Distribution Modes

All environments support `"easy"` and `"hard"` distribution modes. Some environments also support additional modes:

| Mode | Description |
| --- | --- |
| `"easy"` | Reduced difficulty, fewer timesteps to solve |
| `"hard"` | Full difficulty (default) |
| `"extreme"` | Maximum difficulty (game-specific) |
| `"memory"` | Memory-focused variant (game-specific) |
| `"exploration"` | Single fixed level for exploration research |

Environments supporting `"exploration"` mode: CoinRun, CaveFlyer, Leaper, Jumper, Maze, Heist, Climber, Ninja.
