import gymnasium as gym
import numpy as np
from sim_wrapper import OT2Env

# Load your custom environment
env = OT2Env(render=False, max_steps=1000)

# Number of episodes
num_episodes = 5

for episode in range(num_episodes):
    obs, info = env.reset()
    step = 0

    while True:
        # Take a random action from the environment's action space
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Episode: {episode + 1}, Step: {step + 1}, Action: {action}, Reward: {reward}")

        step += 1
        if terminated or truncated:
            print(f"Episode finished after {step} steps. Info: {info}")
            break

env.close()
