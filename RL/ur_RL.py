import time

import gym
import numpy as np
from gym import spaces
from mujoco import *
from mujoco import viewer
from stable_baselines3 import SAC
import torch

print(torch.cuda.is_available())


class UR5eEnv(gym.Env):
    """Custom Environment for UR5e robot arm that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, model_path='./universal_robots_ur5e_RL/scene.xml', target_positions=None):
        super(UR5eEnv, self).__init__()
        # Load the MuJoCo model
        self.model = MjModel.from_xml_path(model_path)
        self.data = MjData(self.model)
        self.viewer = None

        # Define action space and observation space
        self.action_space = spaces.Box(low=-1., high=1., shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        # Targets
        if target_positions is None:
            target_positions = [np.array([0.35, 0, 0.35]), np.array([-0.35, 0, 0.35]), np.array([0, 0.35, 0.35])]
        self.target_positions = target_positions
        self.current_target = 0

    def step(self, action):
        # Execute one time step within the environment
        self.data.ctrl[:] = action
        mj_step(self.model, self.data)
        obs = self._get_obs()

        # Calculate reward
        distance = np.linalg.norm(self.data.xpos[-1,:] - self.target_positions[self.current_target])
        reward = -distance  # Negative reward based on distance to the target

        done = distance < 0.05  # Consider done if close enough to target
        if done:
            self.current_target = (self.current_target + 1) % len(self.target_positions)  # Move to next target

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        mj_resetData(self.model, self.data)
        self.current_target = 0  # Start from the first target
        return self._get_obs()

    def _get_obs(self):
        # Return the current observation
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        # time.sleep(0.1)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Initialize the environment
env = UR5eEnv()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = SAC.load("sac_ur5e")

# Evaluate the model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
