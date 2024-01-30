import time

import mediapy as media
import mujoco.viewer
xml = """
<mujoco model="ur5e scene">

  <include file="ur5e.xml"/>

  <statistic center="0.3 0 0.4" extent="0.8"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
<!--    <geom name="red_box" type="box" size="1 0.02 1" rgba="1 0 0 1"  pos="0.2 0.5 .2"/>-->
  </worldbody>

</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)
mujoco.mj_forward(model, data)
camera = mujoco.MjvCamera()
DURATION = 1000  # seconds
FRAMERATE = 60  # Hz
# model.opt.gravity = (0, 0, -9.8)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while data.time < DURATION:
        # mujoco.mj_step(model, data)
        viewer.sync()
        # time.sleep(1)