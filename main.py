import numpy as np
from typing import Callable, Optional, Union, List
import scipy.linalg
import mediapy as media
import matplotlib.pyplot as plt
import mujoco.viewer
from scipy.spatial.transform import Rotation


def impedance_control():
    k = 30
    b = 10
    m = 1


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

for key in range(model.nkey):
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    pixels = renderer.render()
    media.write_image('image' + str(key) + '.png', renderer.render())
DURATION = 1000  # seconds
FRAMERATE = 60  # Hz

# Initialize to the standing-on-one-leg pose.
mujoco.mj_resetDataKeyframe(model, data, 1)

# Make a new camera, move it to a closer distance.
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance = 2

mujoco.mj_resetDataKeyframe(model, data, 0)

frames = []
jacp = np.zeros((3, 6))
jacr = np.zeros((3, 6))

with mujoco.viewer.launch_passive(model, data) as viewer:
    r = Rotation.from_matrix(data.geom_xmat[-1].reshape(3, 3))
    rpy = r.as_euler('xyz', degrees=False)
    init_pose = np.append(data.geom_xpos[-1], rpy.copy())
    error_array = np.zeros(6)
    k = 20
    b = 10
    m = 1
    dt = 1
    while data.time < DURATION:
        # data.xfrc_applied(mujoco.mjtMouse)
        mujoco.mj_jacBody(model, data, jacp, jacr, 7)
        jacob = np.vstack((jacp, jacr))
        r = Rotation.from_matrix(data.geom_xmat[-1].reshape(3, 3))
        rpy = r.as_euler('xyz', degrees=False)
        error = np.append(data.geom_xpos[-1], rpy) - init_pose
        error_array = np.vstack((error_array, error))

        F = k * error_array[-1] + b * np.diff(error_array, axis=0)[
            -1] / dt  # +m*np.diff(error_array,n=2, axis=0)[-1]/dt
        v = -F
        qvel = jacob.T.dot(v)
        # Step the simulation.
        # data.ctrl = data.qpos + qvel * 0.02

        # print(data.sensor(0).data, data.sensor(1).data)

        mujoco.mj_step(model, data)
        viewer.sync()
        mujoco.mj_printFormattedModel(model, './1.txt', '%f')
        # Render and save frames.
        if len(frames) < data.time * FRAMERATE:
            camera.lookat = data.body('wrist_3_link').subtree_com
            renderer.update_scene(data, camera)
            pixels = renderer.render()
            frames.append(pixels)

# Display video.
video = np.array(frames)
media.write_video('video.mp4', video, fps=FRAMERATE)
