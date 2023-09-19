import mujoco
import mediapy as media
import numpy as np
import matplotlib.pyplot as plt

tippe_top = """
<mujoco model="tippe top">
  <option integrator="RK4"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>

  <worldbody>
    <geom size=".2 .2 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
    <body name="top" pos="0 0 .02">
      <freejoint/>
      <geom name="ball" type="sphere" size=".02" />
      <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008"/>
      <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015"
       contype="0" conaffinity="0" group="3"/>
    </body>
  </worldbody>

  <keyframe>
    <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0 0 0 0 1 200" />
  </keyframe>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(tippe_top)
renderer = mujoco.Renderer(model)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera='closeup')
media.write_image('image.png', renderer.render(), )
print('positions', data.qpos)
print('velocities', data.qvel)
duration = 7
frame_rate = 60
frames = []
mujoco.mj_resetDataKeyframe(model, data, 0)  # Reset the state to keyframe 0
while data.time < duration:
    mujoco.mj_step(model, data)
    if len(frames) < data.time * frame_rate:
        renderer.update_scene(data, "closeup")
        pixels = renderer.render()
        frames.append(pixels)
video = np.array(frames)
media.write_video('video.mp4', video, fps=frame_rate)
timevals = []
angular_velocity = []
stem_height = []

mujoco.mj_resetDataKeyframe(model, data, 0)
while data.time < duration:
    mujoco.mj_step(model, data)
    timevals.append(data.time)
    angular_velocity.append(data.qvel[3:6].copy())
    stem_height.append(data.geom_xpos[2, 2])
dpi = 120
width = 600
height = 800
figsize = (width / dpi, height / dpi)
_, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)

ax[0].plot(timevals, angular_velocity)
ax[0].set_title('angular velocity')
ax[0].set_ylabel('radians / second')

ax[1].plot(timevals, stem_height)
ax[1].set_xlabel('time (seconds)')
ax[1].set_ylabel('meters')
_ = ax[1].set_title('stem height')
plt.show()