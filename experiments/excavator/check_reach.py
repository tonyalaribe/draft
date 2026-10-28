#!/usr/bin/env python3
"""Static checks Draft does not make for an excavator: holding torque per arm joint
with a loaded bucket, and whether the machine tips, across slew angle and arm pose.

    python experiments/excavator/check_reach.py [generated/excavator]
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

out = Path(sys.argv[1] if len(sys.argv) > 1 else "generated/excavator")
P = yaml.safe_load((out / "parameters_resolved.yaml").read_text())
spec = mujoco.MjSpec.from_file(str(out / "excavator.xml"))
load = next(s for s in spec.sites if s.name == "bucket_load")
spec.body("bucket_link").add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0], pos=load.pos,
                                  mass=P["payload_mass"], contype=0, conaffinity=0)
m = spec.compile()
d = mujoco.MjData(m)
arm = ["slew", "boom", "stick", "bucket"]
qadr = {j: m.joint(j).qposadr[0] for j in arm}
dadr = {j: m.joint(j).dofadr[0] for j in arm}
lim = {j: m.jnt_actfrcrange[m.joint(j).id][1] for j in arm}

# Wheel contact patch corners in the chassis frame → support rectangle (root at axle height).
hx = P["wheelbase"] / 2
hy = P["chassis_ly"] / 2 + P["wheel_w"] / 2 + 0.01

def pose(q):
    mujoco.mj_resetData(m, d)
    d.qpos[2] = P["wheel_r"]
    for j, v in q.items():
        d.qpos[qadr[j]] = v
    mujoco.mj_forward(m, d)
    com = d.subtree_com[1]
    # Signed margin to the nearest edge of the support rectangle (negative = tips).
    margin = min(hx - abs(com[0]), hy - abs(com[1]))
    tau = {j: abs(d.qfrc_bias[dadr[j]]) for j in arm[1:]}
    return com, margin, tau, d.site("bucket_teeth").xpos.copy()

print(f"mass {m.body_subtreemass[1]:.1f} kg incl. {P['payload_mass']} kg payload; "
      f"support ±{hx:.2f} x ±{hy:.2f} m\n")
print(f"{'pose':43s}{'reach':>7s}{'CoM x,y':>14s}{'margin':>8s}   " +
      "".join(f"{j:>15s}" for j in arm[1:]))
cases = {
    "level, full reach": dict(boom=0.0, stick=0.0, bucket=0.0),
    "boom up 0.6, stick 0.9 (dig)": dict(boom=0.6, stick=0.9, bucket=0.6),
    "boom up 1.0, stick 1.6 (carry)": dict(boom=1.0, stick=1.6, bucket=1.6),
    "boom -0.4, stick 0.6 (below grade)": dict(boom=-0.4, stick=0.6, bucket=0.5),
}
for slew in (0.0, np.pi / 4, np.pi / 2, np.pi):
    for name, q in cases.items():
        com, margin, tau, tip = pose({**q, "slew": slew})
        reach = np.hypot(*tip[:2])
        t = "".join(f"{tau[j]:7.0f}/{lim[j]:<4.0f}{'!' if tau[j] > lim[j] else ' '}  " for j in arm[1:])
        print(f"slew {np.degrees(slew):3.0f}° {name:34s}{reach:7.2f}{com[0]:7.2f},{com[1]:5.2f}"
              f"{margin:+8.2f}{'  TIPS' if margin < 0 else '      '} {t}")
