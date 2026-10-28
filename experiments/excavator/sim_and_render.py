#!/usr/bin/env python3
"""12 s PD-tracked dig cycle, then drive forward, then three still renders.
`--view` instead loops the cycle live in the MuJoCo viewer (macOS needs `mjpython`).

    python experiments/excavator/sim_and_render.py [generated/excavator]
    mjpython experiments/excavator/sim_and_render.py --view
"""
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

view = "--view" in sys.argv
args = [a for a in sys.argv[1:] if a != "--view"]
out = Path(args[0] if args else "generated/excavator")
m = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
d = mujoco.MjData(m)
arm = ["slew", "boom", "stick", "bucket"]
wheels = [f"{w}_wheel" for w in ("fl", "fr", "rl", "rr")]
qa = {j: m.joint(j).qposadr[0] for j in arm}
da = {j: m.joint(j).dofadr[0] for j in arm + wheels}
lim = {j: m.jnt_actfrcrange[m.joint(j).id][1] for j in arm + wheels}
target = lambda t: dict(slew=0.8 * np.sin(0.4 * t), boom=0.5 + 0.5 * np.sin(0.6 * t),
                        stick=1.0 + 0.8 * np.sin(0.5 * t + 1), bucket=0.6 + 0.8 * np.sin(0.7 * t))

def step():
    t = d.time % 12
    q = target(t)
    for j in arm:
        d.qfrc_applied[da[j]] = np.clip(2000 * (q[j] - d.qpos[qa[j]]) - 200 * d.qvel[da[j]], -lim[j], lim[j])
    for w in wheels:
        d.qfrc_applied[da[w]] = np.clip(-5 * (d.qvel[da[w]] - 2.0 * (1 if d.time % 24 < 12 else -1)),
                                        -lim[w], lim[w]) if t > 8 else 0
    mujoco.mj_step(m, d)


if view:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.time()
            step()
            v.sync()
            time.sleep(max(0, m.opt.timestep - (time.time() - t0)))
    sys.exit()

while d.time < 12:
    step()
    if abs(d.time % 2) < m.opt.timestep:
        tilt = np.degrees(2 * np.arccos(min(1, abs(d.qpos[3]))))
        print(f"t={d.time:5.2f}  base x={d.qpos[0]:.2f} z={d.qpos[2]:.3f}  tilt={tilt:.1f}°")

r = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera()
cam.distance, cam.lookat[:] = 3.2, [0.4, 0, 0.4]
views = {"iso": (135, -20, dict(boom=0.5, stick=1.0, bucket=0.8)),
         "side": (90, -5, {}),
         "dig": (210, -25, dict(slew=0.6, boom=-0.3, stick=1.5, bucket=1.5))}
for name, (cam.azimuth, cam.elevation, q) in views.items():
    mujoco.mj_resetData(m, d)
    for j, v in q.items():
        d.qpos[qa[j]] = v
    mujoco.mj_forward(m, d)
    r.update_scene(d, cam)
    Image.fromarray(r.render()).save(out / f"render_{name}.png")
