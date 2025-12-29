# test_wrapper.py
import numpy as np
from sim_wrapper import OT2Env

if __name__ == "__main__":
    env = OT2Env(render_mode=None)
    obs, info = env.reset()

    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode ended at step {i}, distance={info['distance']:.4f}")
            obs, info = env.reset()

    env.close()
