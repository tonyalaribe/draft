"""Excavator world: the Draft-generated machine on a diggable heightfield.

One environment, three drivers: teleop (`ground_station.py`), the scripted autopilot
(`autopilot.py`) and a learned policy all send the same action
`[slew, boom, stick, bucket, left_drive, right_drive]` in [-1, 1] (fractions of each
joint's no-load speed), at 50 Hz. Joint velocity commands integrate into position targets held
by a torque-clipped PD, so a zero command is a brake.

Soil: the fundamental earthmoving equation F = w(γgd²Nγ + c·d·Nc) resists the teeth
along -v; cells the bucket scoops through are carved into the bucket (capped at its
volume); the fill weighs on `bucket_load`, and it spills onto the terrain once the opening tips
past horizontal. The N-factors (≈1.7) are representative values for a blade at moderate rake (Malaguti, ISARC 2005).

    env = ExcavatorEnv(soil="firm"); obs, _ = env.reset(); obs, r, term, trunc, info = env.step(a)
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml

GEN = Path(__file__).resolve().parents[2] / "generated" / "excavator"
G = 9.81
SOILS = {  # density kg/m³, cohesion Pa
    "loose": dict(rho=1500.0, c=0.0),
    "firm": dict(rho=1700.0, c=5e3),
    "clay": dict(rho=1800.0, c=25e3),
}
N_GAMMA, N_C = 1.7, 1.7
ARM = ["slew", "boom", "stick", "bucket"]
WHEELS = {"fl_wheel": 0, "rl_wheel": 0, "fr_wheel": 1, "rr_wheel": 1}  # → left/right
NO_GROUND = ("bucket_link_geom", "bucket_shell_geom", "stick_link_motor")


def _camera(body, pos, tilt, name):
    s, c = np.sin(tilt), np.cos(tilt)
    body.add_camera(name=name, pos=pos, fovy=75, xyaxes=[0, -1, 0, s, 0, c])


class ExcavatorEnv:
    def __init__(self, soil="loose", arena=8.0, cell=0.05, ctrl_hz=50, gen_dir=GEN,
                 dig_site=(1.05, 0.0), dump_site=(0.0, 1.2), episode_s=60.0, seed=0):
        self.P = yaml.safe_load((Path(gen_dir) / "parameters_resolved.yaml").read_text())
        self.soil = SOILS[soil] if isinstance(soil, str) else soil
        self.dig_site, self.dump_site = np.array(dig_site), np.array(dump_site)
        self.rng = np.random.default_rng(seed)
        spec = mujoco.MjSpec.from_file(str(Path(gen_dir) / "excavator.xml"))
        self.n = int(arena / cell) + 1
        self.cell, self.half = cell, arena / 2
        self.base, self.top = -0.6, 1.2  # surface z = base + data·top; data 0.5 → z 0
        spec.add_hfield(name="terrain", size=[self.half, self.half, self.top, 0.1],
                        nrow=self.n, ncol=self.n, userdata=[0.5] * self.n ** 2)
        spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain",
                                pos=[0, 0, self.base], rgba=[0.55, 0.42, 0.3, 1])
        spec.worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7])
        sites = {s.name: s.pos for s in spec.sites}
        _camera(spec.body("turret_link"), sites["cam_mast"], 0.45, "mast")
        _camera(spec.body("stick_link"), sites["cam_bucket"], 1.1, "bucket")
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
        self.m = m = spec.compile()
        self.d = mujoco.MjData(m)
        m.opt.timestep = 0.002
        self.substeps = int(round(1 / (ctrl_hz * m.opt.timestep)))
        self.max_steps = int(episode_s * ctrl_hz)
        self.hid = m.geom("ground").id
        for g in range(m.ngeom):  # ground bit 2: everything but the digging end collides with it
            name = m.geom(g).name
            if g == self.hid:
                m.geom_contype[g] = m.geom_conaffinity[g] = 2
            elif m.geom_contype[g] and not name.startswith(NO_GROUND):
                m.geom_contype[g] |= 2
                m.geom_conaffinity[g] |= 2
        j = {n: m.joint(n) for n in ARM + list(WHEELS)}
        self.qa = np.array([j[n].qposadr[0] for n in ARM])
        self.da = np.array([j[n].dofadr[0] for n in ARM])
        self.wa = np.array([j[n].dofadr[0] for n in WHEELS])
        self.wqa = np.array([j[n].qposadr[0] for n in WHEELS])
        self.wside = np.array(list(WHEELS.values()))
        lim = lambda n: m.jnt_actfrcrange[j[n].id][1]
        self.tau_max = np.array([lim(n) for n in ARM])
        self.wtau_max = lim("fl_wheel")
        self.vmax = np.array([m.numeric(f"{n}_velocity_limit").data[0] for n in ARM])
        self.wvmax = m.numeric("fl_wheel_velocity_limit").data[0]
        self.qrange = np.array([m.jnt_range[j[n].id] for n in ARM])
        self.kp, self.kd = np.array([2500, 4000, 2500, 800.]), np.array([250, 400, 250, 60.])
        self.bucket = m.body("bucket_link").id
        self.teeth, self.load = m.site("bucket_teeth").id, m.site("bucket_load").id
        self.width = float(self.P["bucket_ly"])
        self.repose = np.tan(np.radians(35))
        self.capacity = self.P["bucket_lx"] * self.P["bucket_ly"] * self.P["bucket_lz"] * 1.2  # heaped
        self.reset()

    # ── terrain ───────────────────────────────────────────────────────────────
    def _ij(self, xy):
        return (np.asarray(xy) + self.half) / self.cell  # (col, row) as floats

    def height(self, xy):
        c, r = self._ij(xy)
        c, r = np.clip(c, 0, self.n - 1.001), np.clip(r, 0, self.n - 1.001)
        c0, r0 = int(c), int(r)
        fc, fr = c - c0, r - r0
        h = self.h
        v = (h[r0, c0] * (1 - fc) * (1 - fr) + h[r0, c0 + 1] * fc * (1 - fr)
             + h[r0 + 1, c0] * (1 - fc) * fr + h[r0 + 1, c0 + 1] * fc * fr)
        return self.base + v * self.top

    def _cells(self, xy, radius):
        c, r = self._ij(xy)
        k = int(np.ceil(radius / self.cell))
        rr, cc = np.mgrid[int(r) - k:int(r) + k + 2, int(c) - k:int(c) + k + 2]
        ok = (rr >= 0) & (rr < self.n) & (cc >= 0) & (cc < self.n)
        rr, cc = rr[ok], cc[ok]
        pts = np.stack([cc, rr], 1) * self.cell - self.half
        return rr, cc, np.linalg.norm(pts - xy, axis=1)

    def _set_terrain(self):
        self.m.hfield_data[:] = self.h.ravel()
        self.terrain_version += 1

    def _make_terrain(self):
        n = self.n
        self.h = np.full((n, n), 0.5)
        x = np.arange(n) * self.cell - self.half
        X, Y = np.meshgrid(x, x)
        dx, dy = self.dig_site
        self.h += 0.18 / self.top * np.exp(-((X - dx) ** 2 + (Y - dy) ** 2) / (2 * 0.25 ** 2))  # spoil heap to dig
        self.zone = np.hypot(X - self.dump_site[0], Y - self.dump_site[1]) < 0.5
        self.h0 = self.h.copy()
        self._set_terrain()

    def dumped_m3(self):
        return float(((self.h - self.h0)[self.zone]).sum() * self.top * self.cell ** 2)

    def _relax(self, xy, radius=0.6, iters=30):
        """Slump slopes steeper than the angle of repose, mass-conserving, near `xy`."""
        c, r = np.round(self._ij(xy)).astype(int)
        k = int(radius / self.cell)
        r0, r1, c0, c1 = max(r - k, 1), min(r + k, self.n - 1), max(c - k, 1), min(c + k, self.n - 1)
        h = self.h[r0 - 1:r1 + 1, c0 - 1:c1 + 1]
        lim = self.repose * self.cell / self.top
        for _ in range(iters):
            moved = False
            for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
                nb = np.roll(h, sh, ax)
                flow = 0.2 * np.maximum(0, h - nb - lim)
                inner = np.zeros_like(flow, bool)
                inner[1:-1, 1:-1] = True
                flow *= inner & np.roll(inner, sh, ax)
                if flow.any():
                    h -= flow
                    h += np.roll(flow, -sh, ax)
                    moved = True
            if not moved:
                break

    # ── gym API ───────────────────────────────────────────────────────────────
    def reset(self, seed=None, qpos=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        self.terrain_version = 0
        self._make_terrain()
        d.qpos[2] = self.P["wheel_r"] + 0.005
        d.qpos[self.qa] = qpos if qpos is not None else [0.0, 0.9, 1.8, 2.0]  # tucked
        mujoco.mj_forward(m, d)
        self.qref = d.qpos[self.qa].copy()
        self.wqref = d.qpos[self.wqa].copy()
        self.fill = 0.0  # m³ in bucket
        self.relax_at = None
        self.steps = 0
        self.stats = dict(dug=0.0, energy_J=0.0, peak_force=0.0,
                          released=0.0, sat_steps=np.zeros(4), min_margin=np.inf, peak_tau=np.zeros(4))
        self.force = 0.0
        return self.obs(), {}

    def obs(self):
        d = self.d
        teeth = d.site_xpos[self.teeth]
        return np.concatenate([
            d.qpos[self.qa], self.qref - d.qpos[self.qa], d.qvel[self.da], d.qfrc_applied[self.da] / self.tau_max,
            d.qpos[3:7], d.qvel[0:6],  # base orientation + twist
            teeth - d.qpos[:3], [teeth[2] - self.height(teeth[:2])],
            [self.fill / self.capacity, self.force / 1000.0],
            np.r_[self.dig_site, self.dump_site] - np.tile(d.qpos[:2], 2),
        ]).astype(np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        m, d = self.m, self.d
        dt = m.opt.timestep * self.substeps
        q = d.qpos[self.qa]
        self.qref = np.clip(self.qref + a[:4] * self.vmax * dt, *self.qrange.T)
        self.qref = np.clip(self.qref, q - 0.25, q + 0.25)  # anti-windup when stalled
        wref = np.where(self.wside == 0, a[4], a[5]) * self.wvmax
        wq = d.qpos[self.wqa]
        self.wqref = np.clip(self.wqref + wref * dt, wq - 0.5, wq + 0.5)
        dumped0 = self.dumped_m3()
        for _ in range(self.substeps):
            tau = np.clip(self.kp * (self.qref - d.qpos[self.qa]) - self.kd * d.qvel[self.da],
                          -self.tau_max, self.tau_max)
            d.qfrc_applied[self.da] = tau
            d.qfrc_applied[self.wa] = np.clip(300 * (self.wqref - d.qpos[self.wqa]) + 30 * (wref - d.qvel[self.wa]),
                                              -self.wtau_max, self.wtau_max)
            self._soil()
            mujoco.mj_step(m, d)
            s = self.stats
            s["energy_J"] += m.opt.timestep * (np.abs(tau * d.qvel[self.da]).sum()
                                               + np.abs(d.qfrc_applied[self.wa] * d.qvel[self.wa]).sum())
            s["sat_steps"] += np.abs(tau) > 0.95 * self.tau_max
            s["peak_tau"] = np.maximum(s["peak_tau"], np.abs(tau))
        self.steps += 1
        self.stats["min_margin"] = min(self.stats["min_margin"], self.tip_margin())
        terminated = bool(self.tilt() > 0.6)
        if self.relax_at is not None:
            self._relax(self.relax_at)
            self._set_terrain()
            self.relax_at = None
        reward = (self.dumped_m3() - dumped0) * self.soil["rho"] - 1e-4 * np.square(a).sum() \
            - 5.0 * terminated
        return self.obs(), reward, terminated, self.steps >= self.max_steps, self.info()

    # ── soil ──────────────────────────────────────────────────────────────────
    def _soil(self):
        m, d = self.m, self.d
        p = d.site_xpos[self.teeth].copy()
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.teeth, vel, 0)
        v = vel[3:]
        speed, vxy = np.linalg.norm(v), np.linalg.norm(v[:2])
        R = d.xmat[self.bucket].reshape(3, 3)
        opening = -R[:, 2]  # bucket opening normal (−z of the bucket frame)
        depth = max(0.0, self.height(p[:2]) - p[2])
        F = np.zeros(3)
        self.force = 0.0
        if depth > 0 and speed > 1e-4:
            full = self.fill >= self.capacity
            mag = self.width * (self.soil["rho"] * G * depth ** 2 * N_GAMMA + self.soil["c"] * depth * N_C)
            mag *= (2.0 if full else 1.0) * np.tanh(speed / 0.02)
            F = -mag * v / speed
            self.force = mag
            self.stats["peak_force"] = max(self.stats["peak_force"], mag)
            if vxy > 0.01 and not full and opening @ (v / speed) > 0.2:
                self._carve(p, v[:2] / vxy)
        # Carried soil weighs on the load point; it spills once the opening tips below horizontal.
        if self.fill > 0:
            if opening[2] < 0.0:
                out = min(self.fill, self.capacity * 2.0 * m.opt.timestep * (0.2 - opening[2]))
                self.fill -= out
                self._deposit(d.site_xpos[self.load][:2], out)
        d.xfrc_applied[self.bucket] = 0
        self._apply(F, p)
        if self.fill > 0:
            self._apply(np.array([0, 0, -self.fill * self.soil["rho"] * G]), d.site_xpos[self.load])

    def _apply(self, F, point):
        d = self.d
        d.xfrc_applied[self.bucket, :3] += F
        d.xfrc_applied[self.bucket, 3:] += np.cross(point - d.xipos[self.bucket], F)

    def _carve(self, p, dir_xy):
        behind = p[:2] - dir_xy * self.cell
        side = np.array([-dir_xy[1], dir_xy[0]])
        cut = (p[2] - self.base) / self.top
        vol = 0.0
        for t in np.linspace(-0.5, 0.5, 5) * self.width:
            c, r = np.round(self._ij(behind + side * t)).astype(int)
            if 0 <= r < self.n and 0 <= c < self.n and self.h[r, c] > cut:
                vol += (self.h[r, c] - cut) * self.top * self.cell ** 2
                self.h[r, c] = cut
        vol = min(vol, self.capacity - self.fill)
        if vol > 0:
            self.fill += vol
            self.stats["dug"] += vol
            self.relax_at = p[:2]
            self._set_terrain()

    def _deposit(self, xy, vol):
        rr, cc, dist = self._cells(xy, 0.15)
        w = np.exp(-(dist / 0.08) ** 2)
        self.h[rr, cc] += vol * w / w.sum() / (self.cell ** 2 * self.top)
        self.stats["released"] += vol
        self.relax_at = xy
        self._set_terrain()

    # ── diagnostics ───────────────────────────────────────────────────────────
    def tilt(self):
        return float(np.arccos(np.clip(self.d.xmat[1][8], -1, 1)))

    def tip_margin(self):
        """Distance inside the wheel rectangle of the whole-machine CoM (incl. carried soil),
        projected along gravity onto the contact plane; negative means it is tipping."""
        d, P = self.d, self.P
        mass = self.m.body_subtreemass[1]
        soil = self.fill * self.soil["rho"]
        com = (d.subtree_com[1] * mass + d.site_xpos[self.load] * soil) / (mass + soil)
        R = d.xmat[1].reshape(3, 3)
        local, g = R.T @ (com - d.qpos[:3]), R.T @ np.array([0, 0, -1.0])
        local = local + g * (-P["wheel_r"] - local[2]) / g[2]
        hx, hy = P["wheelbase"] / 2, P["chassis_ly"] / 2 + P["wheel_w"] / 2 + 0.01
        return min(hx - abs(local[0]), hy - abs(local[1]))

    def info(self):
        s = self.stats
        return dict(fill_kg=self.fill * self.soil["rho"], force_N=self.force,
                    dug_kg=s["dug"] * self.soil["rho"], dumped_kg=self.dumped_m3() * self.soil["rho"],
                    spilled_kg=(s["released"] - self.dumped_m3()) * self.soil["rho"], energy_Wh=s["energy_J"] / 3600,
                    peak_force_N=s["peak_force"], tip_margin=self.tip_margin(),
                    min_tip_margin=s["min_margin"], tilt_deg=np.degrees(self.tilt()), peak_tau_frac=s["peak_tau"] / self.tau_max,
                    sat_frac=s["sat_steps"] / max(1, self.steps * self.substeps), t=self.d.time)
