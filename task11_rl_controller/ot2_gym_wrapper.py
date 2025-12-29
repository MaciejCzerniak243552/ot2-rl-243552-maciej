import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sim_class import Simulation
import pybullet as p

class OT2Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render=False, dt=1/240, max_steps=200, action_limit=5.0, tol=1e-3):
        super().__init__()
        self.render = render
        self.dt = dt
        self.max_steps = max_steps
        self.action_limit = action_limit
        self.tol = tol

        # Your sim (DIRECT if render=False)
        self.sim = Simulation(num_agents=1, render=render, rgb_array=False)

        # ACTION: 3 floats (vx, vy, vz) scaled to [-action_limit, action_limit]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # OBS: [pip_x, pip_y, pip_z, goal_x, goal_y, goal_z]  (6 floats)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.goal = np.zeros(3, dtype=np.float32)
        self.steps = 0

    def _get_pip(self):
        pip = np.array(self.sim.get_pipette_position(robotId=self.sim.robotIds[0]), dtype=np.float32)
        return pip

    def _get_obs(self):
        pip = self._get_pip()
        return np.concatenate([pip, self.goal]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # reset sim
        self.sim.reset(num_agents=1)

        # sample a random goal inside your work envelope:
        x_min, x_max = -0.1871, 0.2531
        y_min, y_max = -0.1706, 0.2195
        z_min, z_max = 0.1694, 0.2896
        gx = self.np_random.uniform(x_min, x_max)
        gy = self.np_random.uniform(y_min, y_max)
        gz = self.np_random.uniform(z_min, z_max)
        self.goal = np.array([gx, gy, gz], dtype=np.float32)

        obs = self._get_obs()
        info = {"goal": self.goal.copy()}
        return obs, info

    def step(self, action):
        self.steps += 1

        # scale action from [-1,1] to [-action_limit, action_limit]
        action = np.array(action, dtype=np.float32)
        u = np.clip(action, -1, 1) * self.action_limit

        # apply as [[x,y,z,drop]]
        self.sim.apply_actions([[float(u[0]), float(u[1]), float(u[2]), 0]])
        p.stepSimulation()

        pip = self._get_pip()
        err = self.goal - pip
        dist = float(np.linalg.norm(err))

        # reward: dense shaping
        reward = -dist
        # success bonus
        terminated = dist < self.tol
        if terminated:
            reward += 10.0

        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        info = {"dist": dist, "pip": pip.copy(), "goal": self.goal.copy()}

        return obs, reward, terminated, truncated, info

    def close(self):
        self.sim.close()
