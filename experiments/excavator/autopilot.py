#!/usr/bin/env python3
"""Scripted dig → curl → swing → dump cycle, driving `ExcavatorEnv` through the same
action vector as teleop and a policy. Also the expert for behaviour cloning.

Arm-plane kinematics (r radial from the slew axis, z up; elevations in rad):
boom e1 = q_boom, stick e2 = e1 - q_stick, bucket e3 = e2 - q_bucket.
The bucket opening is the bucket frame's -z, so e3 = -π/2 digs toward the machine,
e3 ≈ -π carries, and e3 ≈ 0 dumps.

    python experiments/excavator/autopilot.py --soil firm --cycles 5 [--demos demos.npz]
"""
from __future__ import annotations

import argparse

import numpy as np

from env import ExcavatorEnv

DIG, CARRY, DUMP = -np.pi / 2, -2.6, 0.1


class Autopilot:
    def __init__(self, env: ExcavatorEnv, depth=0.10, v_task=0.35, w_slew=0.8):
        self.env, self.depth0, self.v, self.w = env, depth, v_task, w_slew
        m = env.m
        self.L1, self.L2 = env.P["boom_l"], env.P["stick_l"]
        self.tool = m.site_pos[env.teeth][[0, 2]]  # teeth in the bucket frame (x, z)
        self.reset()

    def reset(self):
        self.phase, self.path, self.k, self.stall, self.cycles = "start", [], 0, 0.0, 0
        self.depth, self.wp_time = self.depth0, 0.0
        self.target = None

    # ── kinematics ────────────────────────────────────────────────────────────
    def frame(self):
        d = self.env.d
        p0 = d.xpos[self.env.m.body("boom_link").id]
        yaw = np.arctan2(d.xmat[1][3], d.xmat[1][0])
        return p0, yaw

    def ik(self, r, z, e3):
        """Teeth at (r, z) in the arm plane with bucket elevation e3 → (boom, stick, bucket)."""
        p0, _ = self.frame()
        c, s = np.cos(e3), np.sin(e3)
        tx, tz = self.tool
        wr, wz = r - (tx * c - tz * s), z - (tx * s + tz * c)
        dr, dz = wr, wz - p0[2]
        D = np.clip(np.hypot(dr, dz), abs(self.L1 - self.L2) + 1e-3, 0.995 * (self.L1 + self.L2))
        qs = np.pi - np.arccos(np.clip((self.L1 ** 2 + self.L2 ** 2 - D ** 2) / (2 * self.L1 * self.L2), -1, 1))
        e1 = np.arctan2(dz, dr) + np.arctan2(self.L2 * np.sin(qs), self.L1 + self.L2 * np.cos(qs))
        lo, hi = self.env.qrange[1:].T
        return np.clip([e1, qs, e1 - qs - e3], lo + 0.02, hi - 0.02)

    def pose(self):
        """Measured teeth pose (r, z, e3, slew) from the joint angles."""
        q = self.env.d.qpos[self.env.qa]
        p0, _ = self.frame()
        e2 = q[1] - q[2]
        e3 = e2 - q[3]
        c, s = np.cos(e3), np.sin(e3)
        tx, tz = self.tool
        wr = self.L1 * np.cos(q[1]) + self.L2 * np.cos(e2)
        wz = p0[2] + self.L1 * np.sin(q[1]) + self.L2 * np.sin(e2)
        return np.array([wr + tx * c - tz * s, wz + tx * s + tz * c, e3, q[0]])

    def polar(self, xy):
        p0, yaw = self.frame()
        v = np.asarray(xy) - p0[:2]
        return np.hypot(*v), np.arctan2(v[1], v[0]) - yaw  # radius, slew angle

    # ── plan ──────────────────────────────────────────────────────────────────
    def plan(self, phase):
        env = self.env
        r_dig, th_dig = self.polar(env.dig_site)
        r_dump, th_dump = self.polar(env.dump_site)
        r_dig = min(r_dig, 1.05)
        u = lambda th: np.array([np.cos(th + self.frame()[1]), np.sin(th + self.frame()[1])])
        ground = lambda r, th: env.height(self.frame()[0][:2] + r * u(th))
        a, b = r_dig + 0.10, r_dig - 0.45
        hi = max(ground(a, th_dig), ground(b, th_dig)) + 0.35
        self.phase = phase
        self.path = {  # waypoints (r, z, e3, slew)
            "swing_to_dig": [(0.8, hi + 0.15, CARRY, th_dig)],
            "dig": [(a, ground(a, th_dig) + 0.05, DIG, th_dig),
                    (a, ground(a, th_dig) - self.depth, DIG, th_dig),
                    (b, ground(b, th_dig) - self.depth, DIG - 0.5, th_dig)],
            "curl": [(b, ground(b, th_dig) - 0.02, CARRY, th_dig), (0.8, hi + 0.15, CARRY, th_dig)],
            "swing_to_dump": [(0.8, hi + 0.15, CARRY, th_dump)],
            "dump": [(r_dump - 0.2, hi + 0.2, CARRY, th_dump), (r_dump - 0.2, hi + 0.2, DUMP, th_dump)],
        }[phase]
        self.k, self.wp_time = 0, 0.0
        if self.target is None:
            self.target = self.pose()

    NEXT = {"start": "swing_to_dig", "swing_to_dig": "dig", "dig": "curl", "curl": "swing_to_dump",
            "swing_to_dump": "dump", "dump": "swing_to_dig"}

    PHASES = list(NEXT)[1:]
    TOL = np.array([0.05, 0.05, 0.15, 0.08])

    def goal(self):
        return np.array(self.path[self.k], float)

    def update(self):
        """Advance the plan from the measured pose. Shared by the autopilot and a learned
        controller, which then only has to reach `goal()`. Returns True on reaching a waypoint."""
        env = self.env
        if self.phase == "start":
            self.plan(self.NEXT["start"])
        self.wp_time += env.substeps * env.m.opt.timestep
        reached = bool(np.all(np.abs(self.pose() - self.goal()) < self.TOL))
        last = self.k == len(self.path) - 1
        if (reached or self.wp_time > 4.0) and not (self.phase == "dump" and last and env.fill > 1e-4):
            self.k, self.wp_time = self.k + 1, 0.0
            if self.k >= len(self.path):
                self.cycles += self.phase == "dump"
                self.plan(self.NEXT[self.phase])
        if self.phase == "dig" and env.fill >= 0.98 * env.capacity:
            self.plan("curl")  # full: stop cutting
        return reached

    def features(self):
        """Planner context for a learned controller: phase one-hot, goal − pose, goal."""
        g, p = self.goal(), self.pose()
        err = g - p
        err[3] = (err[3] + np.pi) % (2 * np.pi) - np.pi
        return np.r_[[self.phase == ph for ph in self.PHASES], err, g].astype(np.float32)

    def act(self, update=True):
        env = self.env
        dt = env.substeps * env.m.opt.timestep
        if update:
            self.update()
        goal = self.goal()
        step = np.array([self.v, self.v, 1.2, self.w]) * dt
        q = env.d.qpos[env.qa]
        lag = np.abs(np.r_[self.target[3], self.ik(*self.target[:3])] - q).max()
        if lag < 0.15:  # advance only while the arm keeps up: a force-limited cut
            self.target += np.clip(goal - self.target, -step, step)
            self.stall = 0.0
        else:
            self.stall += dt
            if self.phase == "dig" and self.stall > 0.6:  # stalled in the cut: shallower
                self.depth = max(0.02, self.depth - 0.02)
                self.path = [(r, z + 0.02, e, t) for r, z, e, t in self.path]
                self.stall = 0.0
        self.cmd = np.r_[self.target[3], self.ik(*self.target[:3])]
        a = np.zeros(6)
        a[:4] = np.clip((self.cmd - env.qref) / (env.vmax * dt), -1, 1)
        return a


def run(soil, cycles, demos=None, seconds=240, verbose=True):
    env = ExcavatorEnv(soil=soil, episode_s=seconds)
    ap = Autopilot(env)
    obs, _ = env.reset()
    O, A = [], []
    while ap.cycles < cycles:
        a = ap.act()
        O.append(obs), A.append(a)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    if demos:
        np.savez_compressed(demos, obs=np.array(O), act=np.array(A))
    info["cycles"] = ap.cycles
    info["final_depth"] = ap.depth
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--soil", nargs="+", default=["loose", "firm", "clay"])
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--demos")
    args = ap.parse_args()
    print(f"{'soil':6s}{'cycles':>7s}{'time s':>8s}{'dumped kg':>10s}{'kg/cycle':>9s}{'kg/h':>7s}{'spilt':>7s}"
          f"{'peak N':>8s}{'depth':>7s}{'Wh':>6s}{'tip m':>7s}   peak τ / limit (slew boom stick bucket)   saturated %")
    for s in args.soil:
        i = run(s, args.cycles, args.demos and f"{args.demos}_{s}.npz")
        n = max(i["cycles"], 1)
        print(f"{s:6s}{i['cycles']:7d}{i['t']:8.1f}{i['dumped_kg']:10.1f}{i['dumped_kg'] / n:9.1f}"
              f"{i['dumped_kg'] / i['t'] * 3600:7.0f}{i['spilled_kg']:7.1f}{i['peak_force_N']:8.0f}"
              f"{i['final_depth']:7.2f}{i['energy_Wh']:6.2f}{i['min_tip_margin']:7.2f}   "
              + " ".join(f"{x:4.2f}" for x in i["peak_tau_frac"]) + "   "
              + " ".join(f"{100 * x:4.1f}" for x in i["sat_frac"]))
