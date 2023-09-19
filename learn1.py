import mujoco
import mediapy as media
import matplotlib.pyplot as plt
import time
import mujoco.viewer
import numpy as np
xml = """
<mujoco>
  <worldbody>
  <light name="top" pos="0 0 1"/>
  <body name="box_and_sphere" euler="0 0 -30">
    <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)
print('default gravity', model.opt.gravity)
model.opt.gravity = (0, 0, 10)
print('Total number of DoFs in the model:', model.nv)
print('Generalized positions:', data.qpos)
print('Generalized velocities:', data.qvel)
print('flipped gravity', model.opt.gravity)
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] =True
# mujoco.mj_forward(model, data)
# renderer.update_scene(data)
# model.geom('red_box').rgba[:3] = np.random.rand(3)
# renderer.update_scene(data)
# media.show_image(renderer.render())
# time.sleep(1)

duration = 3.8
frame_rate = 60
frames = []

mujoco.mj_resetData(model,data)
while data.time <duration:
    mujoco.mj_step(model,data)
    if len(frames)<data.time*frame_rate:
        renderer.update_scene(data,scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)
media.show_video(np.array(frames),fps = frame_rate)
video=np.array(frames)
media.write_video('video.mp4', video, fps=frame_rate)

# media.write_image("output.mp4",np.array(frames), fps=frame_rate)
# exit()
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     # start = time.time()
#     while viewer.is_running():
#         step_start = time.time()
#         with viewer.lock():
#             viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
#
#         viewer.sync()
#         time_until_next_step = model.opt.timestep - (time.time() - step_start)
#         if time_until_next_step > 0:
#             time.sleep(time_until_next_step)