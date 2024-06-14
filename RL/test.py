import gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

# 创建环境
env = make_vec_env('Pendulum-v1', n_envs=1)

# 初始化 SAC 模型
model = SAC('MlpPolicy', env, verbose=1)

# 训练模型
model.learn(total_timesteps=50000)

# 保存模型
model.save("sac_pendulum")

# 加载模型
model = SAC.load("sac_pendulum")

# 测试模型
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
