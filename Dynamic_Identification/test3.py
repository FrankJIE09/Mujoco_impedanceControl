from mujoco import *
import numpy as np
from mujoco import viewer

# 加载 MuJoCo 模型
model = MjModel.from_xml_path('./UR5e/scene.xml')
data = MjData(model)
viewer2 = viewer.launch_passive(model, data)

# 仿真步长
dt = 0.01

# 记录数据
num_steps = 1000
qpos = np.zeros((num_steps, model.nq))
qvel = np.zeros((num_steps, model.nv))
qacc = np.zeros((num_steps, model.nv))
torques = np.zeros((num_steps, model.nu))

for i in range(num_steps):
    # 设定控制输入
    ctrl = np.random.uniform(-1, 1, model.nu)  # 示例随机控制输入
    data.ctrl[:] = ctrl

    # 记录数据
    qpos[i, :] = data.qpos[:]
    qvel[i, :] = data.qvel[:]
    qacc[i, :] = (data.qvel[:] - qvel[i-1, :]) / dt if i > 0 else np.zeros(model.nv)
    torques[i, :] = data.qfrc_actuator[:]

    # 进行仿真步
    mj_step(model, data)
    viewer2.sync()

# 保存数据到文件
np.savez('simulation_data.npz', qpos=qpos, qvel=qvel, qacc=qacc, torques=torques)

