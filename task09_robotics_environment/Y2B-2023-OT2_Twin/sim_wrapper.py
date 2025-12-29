import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    def __init__(self, render=False, max_episode_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_episode_steps = max_episode_steps

        # Working envelope bounds determined from the datalab task
        self.env_low = np.array([-0.1871, -0.1706, 0.1694], dtype=np.float32)
        self.env_high = np.array([0.2531, 0.2195, 0.2896], dtype=np.float32)

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
        self.success_reward = 50.0
        self.step_penalty = 0.01
        self.out_of_time_penalty = 50.0

        # keep track of the number of steps
        self.steps = 0
        self.prev_distance = None

    def reset(self, seed=None):
        # being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # set a random goal position for the agent, consisting of x, y, and z coordinates within the working area (you determined these values in the previous datalab task)
        self.goal_position = np.random.uniform(
            low=self.env_low, high=self.env_high
        ).astype(np.float32)

        # Call the environment reset function
        states = self.sim.reset(num_agents=1)

        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        first_key = next(iter(states))
        pipette_position = np.array(
            states[first_key]["pipette_position"], dtype=np.float32
        )
        observation = np.concatenate(
            [pipette_position, self.goal_position]
        ).astype(np.float32)

        # Reset the number of steps
        self.steps = 0
        self.prev_distance = float(np.linalg.norm(self.goal_position - pipette_position))

        info = {
            "distance": self.prev_distance,
            "goal": self.goal_position.copy(),
            "pipette_position": pipette_position.copy(),
        }

        return observation, info

    def step(self, action):
        # Execute one time step within the environment
        # since we are only controlling the pipette position, we accept 3 values for the action and need to append 0 for the drop action
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        action = np.append(action, 0.0)

        # Call the environment step function
        states = self.sim.run([action])  # Why do we need to pass the action as a list? Think about the simulation class.

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
        reward = -distance - self.step_penalty
        self.prev_distance = distance

        # next we need to check if the if the task has been completed and if the episode should be terminated
        # To do this we need to calculate the distance between the pipette position and the goal position and if it is below a certain threshold, we will consider the task complete. 
        # What is a reasonable threshold? Think about the size of the pipette tip and the size of the plants.
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

    def render(self, mode="human"):
        pass
    
    def close(self):
        self.sim.close()
