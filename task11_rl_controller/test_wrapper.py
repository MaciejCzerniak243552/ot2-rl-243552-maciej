from ot2_gym_wrapper import OT2Env

env = OT2Env(render=False)
obs, info = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("OK: ran 1000 random steps")