
import time

import numpy as np
import mujoco.viewer
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from ikpy.chain import Chain

# 从URDF文件中创建机械臂链
my_chain = Chain.from_urdf_file("./config/ur5e.urdf",
                                active_links_mask=[False, True, True, True, True, True, True, False])
#
np.set_printoptions(precision=5, suppress=True, linewidth=100)

# 加载Mujoco模型
model = mujoco.MjModel.from_xml_path("./universal_robots_ur5e_c/scene.xml")
data = mujoco.MjData(model)
model.opt.gravity = (0, 0, -9.8)
mujoco.mj_resetDataKeyframe(model, data, 0)
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance = 0

# 使用Mujoco的viewer来可视化模型
with mujoco.viewer.launch_passive(model, data) as viewer:
    init_orientation = Rotation.from_euler('zyx', [0, 0, 0], degrees=True)
    init_rpy = init_orientation.as_euler('zyx')
    # 定义关键帧的姿态
    quaternions = [Rotation.from_euler('zyx', init_rpy),
                   Rotation.from_euler('zyx', [e + 0.5 if i == 0 else e for i, e in enumerate(init_rpy)], ),
                   Rotation.from_euler('zyx', [e + 0 if i == 0 else e for i, e in enumerate(init_rpy)], ),
                   Rotation.from_euler('zyx', [e + 0.4 if i == 2 else e for i, e in enumerate(init_rpy)], ),
                   Rotation.from_euler('zyx', [e + 0.6 if i == 2 else e for i, e in enumerate(init_rpy)], )]
    key_rots = Rotation.random(5)
    for i in range(quaternions.__len__()):
        key_rots[i] = quaternions[i]

    # 定义关键帧的时间
    key_times = [0, 1, 2, 3, 4]

    # 使用球面插值法生成平滑的插值姿态
    slerp = Slerp(key_times, key_rots)
    times = np.linspace(0, 4, 2000)
    interp_rots = slerp(times)
    i = 0
    init_pos = data.xpos[-1].copy()
    init_matrix = data.xmat[-1].reshape(3, 3).copy()
    target_matrix = np.eye(4)
    # 循环控制机械臂的姿态变化
    while i < 2000:
        # 计算目标变换矩阵
        target_matrix[:3, :3] = np.dot(init_matrix, interp_rots[i].as_matrix())
        target_matrix[:3, 3] = [init_pos[1], -init_pos[0] + 0.1, init_pos[2]]

        # 使用逆运动学求解关节角度
        ik_joint = my_chain.inverse_kinematics_frame(target_matrix,
                                                     initial_position=np.append(np.append(np.array(0), data.qpos),
                                                                                np.array(0)), orientation_mode='all')

        # 计算初始姿态与舵机的关系
        A = my_chain.forward_kinematics(np.append(np.append(np.array(0), data.qpos), np.array(0)))
        init_orientation.as_matrix().T.dot(A[:3, :3])
        B = my_chain.forward_kinematics(ik_joint)

        # 计算实际位置
        actual_pos = np.eye(4)
        actual_pos[:3, :3] = data.xmat[-1].reshape(3, 3)
        actual_pos[:3, 3] = [data.xpos[-1][1], -data.xpos[-1][0], data.xpos[-1][2]]

        # 控制舵机运动
        data.ctrl = ik_joint[1:7]
        mujoco.mj_step(model, data)
        # time.sleep(0.01)
        i = i + 1
        viewer.sync()
