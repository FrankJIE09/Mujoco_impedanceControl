import time

import numpy as np
from typing import Callable, Optional, Union, List
import scipy.linalg
import mediapy as media
import matplotlib.pyplot as plt
import mujoco.viewer
from scipy.spatial.transform import Rotation

np.set_printoptions(precision=9, suppress=True, linewidth=100)


def jug_is_inv(matrix):
    det_matrix = np.linalg.det(matrix)

    if det_matrix != 0:
        return True
    else:
        return False


def impedance_control(Fxe, vk, x):
    k = [30, 30, 30, 10, 10, 10]
    k = np.zeros_like(k)
    b = [10, 10, 10, 1, 1, 1]
    m = 1
    FPlusDF = Fxe
    dt = 0.02
    dvkPlus = ((FPlusDF - b * vk - k * x) / m)
    vkPlus = dt * dvkPlus + vk
    return vkPlus


# More legible printing from numpy.
model = mujoco.MjModel.from_xml_path("./ur10e_mujoco/scene.xml")
data = mujoco.MjData(model)
# model.opt.gravity = (0, 0, -9.8)
model.opt.timestep = 0.002
DURATION = 1000  # seconds
FRAMERATE = 60  # Hz

mujoco.mj_resetDataKeyframe(model, data, 0)

jacp = np.zeros((3, 6))
jacr = np.zeros((3, 6))
xe_array = np.empty((0, 6))
xd_array = np.empty((0, 6))

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Initpose
    Init_pos = np.eye(4)
    Init_pos[:3, :3] = data.ximat[-1].reshape(3, 3)
    Init_pos[:3, 3] = [data.xipos[-1][0], data.xipos[-1][1], data.xipos[-1][2]]
    r = Rotation.from_matrix(Init_pos[:3, :3])
    rpy = r.as_euler('xyz', degrees=False)
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    xd_matrix = np.zeros(6)
    xd_matrix[0] = Init_pos[0, 3].copy()  # x
    xd_matrix[1] = Init_pos[1, 3].copy()  # y
    xd_matrix[2] = Init_pos[2, 3].copy()  # z
    xd_matrix[3] = roll.copy()
    xd_matrix[4] = pitch.copy()
    xd_matrix[5] = yaw.copy()
    i = 0
    data.qacc = np.zeros(6)
    data.qvel = np.zeros(6)
    loop = 0
    dt = 0.002

    while data.time < DURATION:
        # data.xfrc_applied(mujoco.mjtMouse)
        mujoco.mj_jacBody(model, data, jacp, jacr, 7)
        J = np.vstack((jacp, jacr))
        if i == 0:
            J_last = J
            i = 1
        # mujoco.mj_printFormattedData(model)
        if loop < 0:
            data.xfrc_applied[-1] = np.array([0, 1, 0, 0, 0, 0])
            Fext = data.xfrc_applied[-1]
            loop = loop + 1
        else:
            # data.xfrc_applied[-1] = np.array([0, 0, 0, 0, 0, 0])
            Fext = data.xfrc_applied[-1]
        Md = np.eye(6) * 1000
        Md[3:, 3:] = Md[3:, 3:]
        Bd = 10 * Md
        Kd = 30 * Md
        # Bd[3:, 3:] = Bd[3:, 3:]  * 0
        # Kd[3:, 3:] = Kd[3:, 3:]  * 0

        Md_inv = np.linalg.inv(Md)
        J_dot = (J - J_last) / dt
        J_last = J.copy()
        if jug_is_inv(J):
            J_inv = np.linalg.inv(J)
        else:
            print("J 奇异")
            J_inv = np.linalg.pinv(J)
        # 计算实际位置
        actual_pos = np.eye(4)
        actual_pos[:3, :3] = data.ximat[-1].reshape(3, 3)
        actual_pos[:3, 3] = [data.xipos[-1][0], data.xipos[-1][1], data.xipos[-1][2]]
        r = Rotation.from_matrix(actual_pos[:3, :3])
        rpy = r.as_euler('xyz', degrees=False)
        roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
        xr_matrix = np.zeros(6)
        xr_matrix[0] = actual_pos[0, 3]  # x
        xr_matrix[1] = actual_pos[1, 3]  # y
        xr_matrix[2] = actual_pos[2, 3]  # z
        xr_matrix[3] = roll
        xr_matrix[4] = pitch
        xr_matrix[5] = yaw
        xd_array = np.vstack((xd_array, xd_matrix))
        xd_dot_array = np.diff(xd_array, axis=0) / dt
        xd_dot_dot_array = np.diff(xd_dot_array, axis=0) / dt

        xe_matrix = xr_matrix - xd_matrix

        xe_array = np.vstack((xe_array, xe_matrix))
        xe_dot_array = np.diff(xe_array, axis=0) / dt
        xe_dot_dot_array = np.diff(xe_dot_array, axis=0) / dt
        if xe_dot_dot_array.size == 0:
            continue
        if xd_dot_dot_array.size == 0:
            continue
        xe = xe_array[-1]
        xe_dot = xe_dot_array[-1]
        xe_dot_dot = xe_dot_dot_array[-1]

        xd_dot_dot = xd_dot_dot_array[-1]

        # data.qacc = J_inv @ Md_inv @ (Bd @ xe_dot + Kd @ xe + Md @ xd_dot_dot - Md @ J_dot @ data.qvel - Fext)
        data.qacc = J_inv @ Md_inv @ (-Bd @ xe_dot - Kd @ xe + Md @ xd_dot_dot - Md @ J_dot @ data.qvel - Fext)

        # data.qvel = data.qvel + data.qacc * dt
        # data.qpos = data.qpos + data.qvel * dt
        print(data.qacc)

        # print(data.qacc)
        # data.qacc = np.array([0,0,0,0,0,0])/100
        mujoco.mj_inverse(model, data)
        data.ctrl = data.qfrc_inverse - np.transpose(J) @ Fext
        mujoco.mj_step(model, data)
        viewer.sync()
