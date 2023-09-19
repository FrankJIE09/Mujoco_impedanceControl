import numpy as np
from typing import Callable, Optional, Union, List
import scipy.linalg
import mediapy as media
import matplotlib.pyplot as plt
import mujoco

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)
with open('./mujoco/model/humanoid/humanoid.xml', 'r') as f:
    xml = f.read()
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)
mujoco.mj_forward(model, data)
renderer.update_scene(data)
media.write_image('image.png', renderer.render())
frames = []

for key in range(model.nkey):
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    pixels = renderer.render()
    media.write_image('image' + str(key) + '.png', renderer.render())
DURATION  = 3   # seconds
FRAMERATE = 60  # Hz

# Initialize to the standing-on-one-leg pose.
mujoco.mj_resetDataKeyframe(model, data, 1)

frames = []
while data.time < DURATION:
  # Step the simulation.
  mujoco.mj_step(model, data)

  # Render and save frames.
  if len(frames) < data.time * FRAMERATE:
    renderer.update_scene(data)
    pixels = renderer.render()
    frames.append(pixels)


# Display video.
video = np.array(frames)
media.write_video('video2.mp4', video, fps=FRAMERATE)
