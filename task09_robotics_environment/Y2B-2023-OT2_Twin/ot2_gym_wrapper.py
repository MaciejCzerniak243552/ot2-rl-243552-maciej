import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sim_class import Simulation


class OT2Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 240}

    def __init__(self, render_mode=None, max_episode_steps=1000, max_steps=None):
        super().__init__()

        # Support legacy boolean render flag
        if isinstance(render_mode, bool):
            render_mode = "human" if render_mode else None
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode={render_mode}")

        self.render_mode = render_mode
        self.max_episode_steps = max_steps if max_steps is not None else max_episode_steps

        # Working envelope bounds determined from the datalab task
        self.env_low = np.array([-0.1871, -0.1706, 0.2294], dtype=np.float32)
        self.env_high = np.array([0.2531, 0.2196, 0.3495], dtype=np.float32)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        obs_low = np.concatenate([self.env_low, self.env_low]).astype(np.float32)
        obs_high = np.concatenate([self.env_high, self.env_high]).astype(np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Reward/termination parameters
        self.success_threshold = 0.001  # 1 mm tolerance
        self.success_reward = 100.0
        self.step_penalty = 0.01
        self.out_of_time_penalty = 50.0

        self.steps = 0
        self.prev_distance = None
        self.goal_position = None

        render_gui = self.render_mode == "human"
        rgb_array = self.render_mode == "rgb_array"
        self.sim = Simulation(num_agents=1, render=render_gui, rgb_array=rgb_array)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Sample a random goal within the workspace
        self.goal_position = self.np_random.uniform(low=self.env_low, high=self.env_high).astype(np.float32)

        # Reset simulation state
        states = self.sim.reset(num_agents=1)

        first_key = next(iter(states))
        pipette_position = np.array(states[first_key]["pipette_position"], dtype=np.float32)
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

        self.steps = 0
        self.prev_distance = float(np.linalg.norm(self.goal_position - pipette_position))

        info = {
            "distance": self.prev_distance,
            "goal": self.goal_position.copy(),
            "pipette_position": pipette_position.copy(),
        }

        return observation, info

    def step(self, action):
        # Clip and keep float32 actions, append drop flag expected by Simulation
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        action_cmd = np.concatenate([action, np.array([0.0], dtype=np.float32)])

        states = self.sim.run([action_cmd])

        first_key = next(iter(states))
        pipette_position = np.array(states[first_key]["pipette_position"], dtype=np.float32)
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

        distance = float(np.linalg.norm(self.goal_position - pipette_position))
        reward = -distance - self.step_penalty
        self.prev_distance = distance

        if distance < 0.1:
            reward += 1
        if distance < 0.05:
            reward += 5
        if distance <= 0.005:
            reward += 10
        terminated = distance < self.success_threshold
        if terminated:
            reward += self.success_reward + ((self.max_episode_steps - self.steps) * 10.1)

        truncated = (self.steps + 1) >= self.max_episode_steps
        if truncated:
            reward -= self.out_of_time_penalty

        info = {
            "distance": distance,
            "goal": self.goal_position.copy(),
            "pipette_position": pipette_position.copy(),
        }

        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            frame = getattr(self.sim, "current_frame", None)
            return np.array(frame) if frame is not None else None
        # For "human", PyBullet GUI is already running; nothing to return
        return None
    
    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
