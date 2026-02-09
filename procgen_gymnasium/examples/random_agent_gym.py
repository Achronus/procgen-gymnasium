"""
Example random agent script using the Gymnasium API to demonstrate that procgen works
"""

import gymnasium as gym
import numpy as np

# This import triggers environment registration
import procgen_gymnasium  # noqa: F401

env = gym.make_vec("procgen_gym/procgen-coinrun-v0", num_envs=1)
obs, info = env.reset()
step = 0

while True:
    action = np.array([env.action_space.sample()])
    obs, rew, terminated, truncated, info = env.step(action)
    print(f"step {step} reward {rew[0]} terminated {terminated[0]}")
    step += 1
    if terminated[0] or truncated[0]:
        break

env.close()
