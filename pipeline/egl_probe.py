"""Print the first EGL device on which MuJoCo can render offscreen (PBS/Slurm often set CUDA_VISIBLE_DEVICES to GPU
UUIDs, which robosuite cannot map to an EGL device). Exit code 1 if none works."""
import os
import subprocess
import sys

from mujoco.egl import egl_ext as EGL

n = len(EGL.eglQueryDevicesEXT())
code = ("import mujoco; m=mujoco.MjModel.from_xml_string(\"<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>\"); "
        "r=mujoco.Renderer(m,32,32); r.update_scene(mujoco.MjData(m)); r.render()")
for i in range(n):
    env = dict(os.environ, MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID=str(i))
    if subprocess.run([sys.executable, '-c', code], env=env, capture_output=True).returncode == 0:
        print(i)
        sys.exit(0)
sys.exit(1)
