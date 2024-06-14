import time
import gym
import numpy as np
from gym import spaces
from mujoco import MjModel, MjData, mj_step, mj_resetData, viewer
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import torch
import zipfile
import os

print(torch.cuda.is_available())


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model.zip')

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.log_dir, f'model_{self.n_calls}.zip')
            self.model.save(model_path)
            with zipfile.ZipFile(os.path.join(self.log_dir, f'model_{self.n_calls}_archive.zip'), 'w') as zipf:
                zipf.write(model_path, os.path.basename(model_path))
            os.remove(model_path)
        return True


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
        done = False
        # Execute one time step within the environment
        self.data.ctrl[:] = action
        mj_step(self.model, self.data)
        obs = self._get_obs()

        # Calculate reward
        distance = np.linalg.norm(self.data.xpos[-1, :] - self.target_positions[self.current_target])
        reward = -distance  # Negative reward based on distance to the target

        # Check if the current target is reached
        point_done = distance < 0.05  # Consider done if close enough to target
        if point_done:
            print("Reached target", self.current_target)
            self.current_target += 1  # Move to next target
            if self.current_target >= len(self.target_positions):
                done = True  # End episode if all targets are reached
                self.current_target = 0  # Reset to the first target for the next episode
            reward += 100 * self.current_target  # Add bonus for reaching the target

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

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Initialize the environment
env = UR5eEnv()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SAC("MlpPolicy", env=env, device=device, verbose=1)

# Create log dir
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Initialize the callback
callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)

# Train the agent
model.learn(total_timesteps=500000, callback=callback)

# Save the final model
model.save("sac_ur5e_final")
