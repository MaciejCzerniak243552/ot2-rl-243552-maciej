# ot2_gym_wrapper.py

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces

from sim_class import Simulation


class OT2Env(gym.Env):
    """
    Gymnasium-compatible wrapper around the OT-2 PyBullet Simulation.

    Task:
        Move the pipette tip to a randomly sampled 3D target position inside a
        predefined workspace (work envelope).

    Agent:
        Continuous control over pipette joint velocities in x, y, z.

    Observation (9D):
        [pipette_x, pipette_y, pipette_z,
         target_x,   target_y,   target_z,
         dx,         dy,         dz]

        where d* = target - pipette

    Action (3D):
        [vx_cmd, vy_cmd, vz_cmd] in [-1, 1],
        internally scaled to joint velocity commands.

    Episode termination:
        - Success: distance to target < success_threshold
        - Out of bounds: pipette leaves workspace
        - Timeout: max_episode_steps reached (truncation)
    """

    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "render_fps": 240,
    }

    def __init__(
        self,
        # render = False,
        render_mode: Optional[str] = None,
        max_episode_steps=1000,
        num_agents: int = 1,
    ):
        super().__init__()

        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode={render_mode}")

        # self.render = render
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.num_agents = num_agents

        # ----- Workspace definition (tune as needed) -----
        # Approximate 3D work envelope of the pipette in world coordinates.
        # These limits are used to:
        #   - Sample targets
        #   - Check for out-of-bounds
        #   - Define observation_space bounds
        self.env_low = np.array([-0.1871, -0.1706, 0.1694], dtype=np.float32)
        self.env_high = np.array([0.2531, 0.2195, 0.2896], dtype=np.float32)

        # ----- Action space -----
        # RL actions are normalized in [-1, 1] and scaled to joint velocities.
        # self.max_xy_velocity = 0.3  # tune based on stability & realism
        # self.max_z_velocity = 0.3

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # ----- Observation space -----
        # pipette, target, and difference (target - pipette)
        obs_low_pipette = self.env_low
        obs_high_pipette = self.env_high

        obs_low_target = self.env_low
        obs_high_target = self.env_high

        diff_low = self.env_low - self.env_high  # min possible diff
        diff_high = self.env_high - self.env_low  # max possible diff

        obs_low = np.concatenate(
            [obs_low_pipette, obs_low_target, diff_low] #diff_low
        ).astype(np.float32)
        obs_high = np.concatenate(
            [obs_high_pipette, obs_high_target, diff_high] #diff_high
        ).astype(np.float32)

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # ----- Reward / termination parameters -----
        self.success_threshold = 0.001  # 1 mm
        self.success_reward = 50.0
        self.step_penalty = 0.01
        self.out_of_time_penalty = 50.0
        self.out_of_bounds_penalty = 100.0
        self.distance_scale = 1.0  # reward = -distance_scale * distance

        # ----- Internal state -----
        self.sim: Optional[Simulation] = None
        self.current_pipette_pos: Optional[np.ndarray] = None
        self.target_pos: Optional[np.ndarray] = None
        self.current_step: int = 0

        # Initialize Simulation
        self._create_sim()

    # --------------------------------------------------------------------- #
    # Core Gymnasium API
    # --------------------------------------------------------------------- #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and start a new episode.

        Returns:
            observation (np.ndarray), info (dict)
        """
        super().reset(seed=seed)

        # Reset PyBullet simulation
        if self.sim is not None:
            # re-use existing Simulation instance
            states = self.sim.reset(num_agents=self.num_agents)
        else:
            # if somehow missing, recreate
            self._create_sim()
            states = self.sim.reset(num_agents=self.num_agents)
        
        self.prev_dist = None

        # Get pipette position for the single robot
        self.current_pipette_pos = self._extract_pipette_pos(states)

        # Sample a random target position in the workspace
        self.target_pos = self._sample_position_inside_workspace()

        self.current_step = 0

        obs = self._get_obs()
        info = {
            "target": self.target_pos.copy(),
            "pipette_position": self.current_pipette_pos.copy(),
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take one environment step.

        Args:
            action: np.ndarray of shape (3,) in [-1, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip to action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Simulation expects: [[vx, vy, vz, drop_flag], ...] per robot
        # We ignore droplet behavior for the reaching task (drop_flag=0).
        actions_list = [[action[0], action[1], action[2], 0.0]]

        states = self.sim.run(actions_list, num_steps=1)

        self.current_pipette_pos = self._extract_pipette_pos(states)
        self.current_step += 1

        # Compute distance to target
        distance = float(
            np.linalg.norm(self.target_pos - self.current_pipette_pos)
        )

        if self.prev_dist is None:
            reward = -self.distance_scale * distance
        else:
            # Base reward: negative distance and step penalty
            reward = self.distance_scale * (self.prev_dist - distance) - self.step_penalty

        self.prev_dist = distance

        terminated = False
        truncated = False

        # Success condition
        if distance < self.success_threshold:
            terminated = True
            reward += self.success_reward

        # # Out-of-bounds penalty / termination
        # if not self._inside_workspace(self.current_pipette_pos):
        #     terminated = True
        #     reward -= self.out_of_bounds_penalty
        
        # Time limit
        if self.current_step >= self.max_episode_steps:
            truncated = True
            reward -= self.out_of_time_penalty

        obs = self._get_obs()
        info = {
            "distance": distance,
            "target": self.target_pos.copy(),
            "pipette_position": self.current_pipette_pos.copy(),
            "is_success": distance < self.success_threshold,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment.

        - 'human': PyBullet GUI is handled by Simulation (p.GUI)
        - 'rgb_array': returns the last camera frame as np.ndarray (H, W, 4)
        """
        if self.render_mode == "human":
            # Nothing specific to do: PyBullet GUI is already active.
            return None

        if self.render_mode == "rgb_array":
            # Simulation stores current frame when rgb_array=True
            frame = getattr(self.sim, "current_frame", None)
            # PyBullet returns an image as a flat array; often already in
            # shape (H, W, 4). We just return it as-is.
            if frame is None:
                return None
            return np.asarray(frame)

        # If render_mode is None, do nothing
        return None

    def close(self):
        """Close the underlying PyBullet simulation."""
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _create_sim(self):
        """Create the underlying Simulation instance."""
        # For 'human' rendering, use PyBullet GUI, otherwise DIRECT mode.
        render_flag = self.render_mode == "human"
        rgb_flag = self.render_mode == "rgb_array"

        self.sim = Simulation(
            num_agents=self.num_agents, render=render_flag, rgb_array=rgb_flag
        )

    def _extract_pipette_pos(self, states: Dict[str, Any]) -> np.ndarray:
        """
        Extract the pipette position for the single robot from Simulation.get_states().
        """
        # grab the first entry
        first_key = next(iter(states.keys()))
        pipette_pos = states[first_key]["pipette_position"]
        return np.asarray(pipette_pos, dtype=np.float32)

    def _sample_position_inside_workspace(self) -> np.ndarray:
        """Uniformly sample a 3D point inside the workspace box."""
        return self.np_random.uniform(
            low=self.env_low, high=self.env_high
        ).astype(np.float32)

    # def _inside_workspace(self, pos: np.ndarray) -> bool:
    #     """Check if a 3D point is inside the workspace bounds."""
    #     return bool(
    #         np.all(pos >= self.env_low)
    #         and np.all(pos <= self.env_high)
    #     )

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector."""
        pipette = self.current_pipette_pos
        target = self.target_pos
        diff = target - pipette
        obs = np.concatenate([pipette, target, diff]).astype(np.float32)
        return obs


# Optional: a small factory function for SB3-style env creation
def make_ot2_env(render_mode: Optional[str] = None) -> OT2Env:
    """
    Convenience function to create the OT2Env, useful when
    passing to Stable Baselines 3's make_vec_env or similar utilities.
    """
    return OT2Env(render_mode=render_mode)