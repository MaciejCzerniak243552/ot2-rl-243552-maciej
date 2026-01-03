import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    def __init__(self, render: bool = False, max_episode_steps: int = 1000):
        super().__init__()
        self.render_mode = render
        self.max_episode_steps = max_episode_steps
        # Apply several Bullet steps per RL step so the pipette actually travels
        self.frame_skip = 10
        # Curriculum: start with nearby goals for first episodes
        self.curriculum_resets = 200
        self.curriculum_radius = 0.05  # 5 cm radius around current pipette
        # Keep some easy goals always (fraction of resets)
        self.easy_goal_fraction = 0.25
        self.easy_goal_radius = 0.05  # 5 cm

        # Working envelope bounds determined from the datalab task
        self.env_low = np.array([-0.1871, -0.1706, 0.1694], dtype=np.float32)
        self.env_high = np.array([0.2531, 0.2195, 0.2896], dtype=np.float32)
        # Scale raw actions ([-1,1]) to meaningful joint velocities (m/s-ish)
        self.velocity_scale = np.array([0.24, 0.24, 0.12], dtype=np.float32)
        # Finer scaling when close to goal
        self.near_goal_radius = 0.02  # 2 cm
        self.near_goal_velocity_scale = np.array([0.12, 0.12, 0.06], dtype=np.float32)

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        obs_low = np.concatenate([self.env_low, self.env_low]).astype(np.float32)
        obs_high = np.concatenate([self.env_high, self.env_high]).astype(np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Reward/termination parameters
        self.success_threshold = 0.001  # 1 mm tolerance
        self.success_reward = 220.0
        self.step_penalty = 0.01
        self.out_of_time_penalty = 50.0

        # keep track of the number of steps
        self.steps = 0
        self.prev_distance = None
        self.goal_position = None
        self.episode_count = 0

    def reset(self, seed=None, options=None):
        # being able to set a seed is required for reproducibility
        super().reset(seed=seed)

        # Use env-specific RNG if available, otherwise fall back to global numpy
        rng = getattr(self, "np_random", np.random)

        # Call the environment reset function
        states = self.sim.reset(num_agents=1)

        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        first_key = next(iter(states))
        pipette_position = np.array(
            states[first_key]["pipette_position"], dtype=np.float32
        )

        # Curriculum / easy-goal sampling
        use_curriculum = self.episode_count < self.curriculum_resets
        use_easy = rng.random() < self.easy_goal_fraction
        if use_curriculum or use_easy:
            radius = self.curriculum_radius if use_curriculum else self.easy_goal_radius
            offset = rng.uniform(-radius, radius, size=3)
            self.goal_position = np.clip(
                pipette_position + offset, self.env_low, self.env_high
            ).astype(np.float32)
        else:
            self.goal_position = rng.uniform(
                low=self.env_low, high=self.env_high
            ).astype(np.float32)

        observation = np.concatenate(
            [pipette_position, self.goal_position, self.prev_distance]
        ).astype(np.float32)

        # Reset the number of steps
        self.steps = 0
        self.episode_count += 1
        self.prev_distance = float(np.linalg.norm(self.goal_position - pipette_position))

        info = {
            "distance": self.prev_distance,
            "goal": self.goal_position.copy(),
            "pipette_position": pipette_position.copy(),
        }

        return observation, info

    def step(self, action):
        # Execute one time step within the environment
        # Ensure action is in the valid range and correct dtype
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Scale to usable velocities
        if self.prev_distance is not None and self.prev_distance < self.near_goal_radius:
            action[:3] = action[:3] * self.near_goal_velocity_scale
        else:
            action[:3] = action[:3] * self.velocity_scale

        # Append 0.0 for the "drop" action dimension used by the simulation
        action = np.append(action, 0.0)

        # Call the environment step function
        states = self.sim.run([action], num_steps=self.frame_skip)

        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        first_key = next(iter(states))
        pipette_position = np.array(
            states[first_key]["pipette_position"], dtype=np.float32
        )
        observation = np.concatenate(
            [pipette_position, self.goal_position]
        ).astype(np.float32)

        # Calculate the reward, this is something that you will need to experiment with to get the best results
        distance = float(np.linalg.norm(self.goal_position - pipette_position))
        progress = self.prev_distance - distance
        # Dense shaping: reward progress strongly, pull toward target, penalize time
        reward = 100.0 * progress - 0.3 * distance - self.step_penalty
        # Fine accuracy bonus to help lock in sub-centimeter behavior
        if distance < 0.01:
            reward += 2.0  # extra pull inside 1 cm (in addition to below)
        self.prev_distance = distance

        # next we need to check if the if the task has been completed and if the episode should be terminated
        # To do this we need to calculate the distance between the pipette position and the goal position and if it is below a certain threshold, we will consider the task complete. 
        # What is a reasonable threshold? Think about the size of the pipette tip and the size of the plants.
        # Intermediate bonuses aligned with points table to guide fine accuracy
        if distance < 0.01:
            reward += 4.0
        if distance < 0.005:
            reward += 6.0
        if distance < self.success_threshold:
            terminated = True
            # we can also give the agent a positive reward for completing the task
            reward += self.success_reward
        else:
            terminated = False

        # next we need to check if the episode should be truncated, we can check if the current number of steps is greater than the maximum number of steps
        if (self.steps + 1) >= self.max_episode_steps:
            truncated = True
            reward -= self.out_of_time_penalty
        else:
            truncated = False

        info = {
            "distance": distance,
            "goal": self.goal_position.copy(),
            "pipette_position": pipette_position.copy(),
        }  # we don't need to return any additional information

        # increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        # Hook for rendering
        if self.render_mode and hasattr(self.sim, "render"):
            self.sim.render()

    def close(self):
        if hasattr(self.sim, "close"):
            self.sim.close()
        super().close()
