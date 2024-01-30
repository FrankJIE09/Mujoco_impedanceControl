import numpy as np
from typing import Callable, Optional, Union, List
import scipy.linalg
import mediapy as media
import matplotlib.pyplot as plt
import mujoco.viewer
from scipy.spatial.transform import Rotation


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
np.set_printoptions(precision=5, suppress=True, linewidth=100)
model = mujoco.MjModel.from_xml_path("./universal_robots_ur5e/scene.xml")

data = mujoco.MjData(model)
# mujoco.mj_jacSubtreeCom(model,data,None,0)
renderer = mujoco.Renderer(model)
mujoco.mj_forward(model, data)
renderer.update_scene(data)
media.write_image('image.png', renderer.render())
frames = []
model.opt.gravity = (0, 0, -9.8)
DURATION = 1000  # seconds
FRAMERATE = 60  # Hz

# Initialize to the standing-on-one-leg pose.
mujoco.mj_resetDataKeyframe(model, data, 1)

# Make a new camera, move it to a closer distance.
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance = 2

mujoco.mj_resetDataKeyframe(model, data, 0)

jacp = np.zeros((3, 6))
jacr = np.zeros((3, 6))

with mujoco.viewer.launch_passive(model, data) as viewer:
    r = Rotation.from_matrix(data.geom_xmat[-1].reshape(3, 3))
    rpy = r.as_euler('xyz', degrees=False)
    init_pose = np.append(data.geom_xpos[-1], rpy.copy())
    error_array = np.zeros(6)
    k = 30
    b = 10
    m = 1
    f_init = np.append(data.sensor(0).data, data.sensor(1).data)
    while data.time < DURATION:
        # data.xfrc_applied(mujoco.mjtMouse)
        mujoco.mj_jacBody(model, data, jacp, jacr, 7)
        jacob = np.vstack((jacp, jacr))
        r = Rotation.from_matrix(data.geom_xmat[-1].reshape(3, 3))
        rpy = r.as_euler('xyz', degrees=False)
        error = np.append(data.geom_xpos[-1], rpy) - init_pose
        error_array = np.vstack((error_array, error))
        # mujoco.mj_printFormattedData(model)
        f = np.append(data.sensor(0).data, data.sensor(1).data)
        print(f)
        v = impedance_control(f, vk=np.diff(error_array, axis=0)[
            -1], x=error_array[-1])
        qvel = jacob.T.dot(v)
        # Step the simulation.
        data.ctrl = data.qpos + qvel * 0.02
        print(data.ctrl)
        mujoco.mj_step(model, data)
        viewer.sync()
        # Render and save frames.
