#!/usr/bin/env python3
"""Remote-operation ground station: chase view, mast FPV and bucket camera, HUD, and
MANUAL / AUTO / POLICY modes over the one `ExcavatorEnv` action interface.

    python experiments/excavator/ground_station.py [--soil firm] [--policy runs/x/model.zip]
    python experiments/excavator/ground_station.py --calibrate      # print transmitter axes live

Keyboard (ISO excavator pattern): W/S stick out/in, A/D swing, ↑/↓ boom down/up,
←/→ bucket curl/dump; I/K drive, J/L turn; Tab mode, Space e-stop, Backspace reset.
Mouse: drag to orbit the chase view, wheel to zoom.

A drone transmitter in USB-joystick mode (EdgeTX/OpenTX: "USB Joystick") shows up as
an SDL joystick. Map its axes in AXES; DRIVE_SWITCH flips the right stick between boom/bucket
and the wheels. If the link drops in MANUAL, every command goes to zero and the brakes hold.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np
import pygame

from autopilot import Autopilot
from train import load_policy
from env import ARM, ExcavatorEnv

# Transmitter axis → (command, sign). Mode 2: left stick = throttle/yaw, right = pitch/roll.
AXES = {3: ("slew", 1), 2: ("stick", -1), 1: ("boom", 1), 0: ("bucket", -1)}
DRIVE_SWITCH = 4  # aux axis > 0 → right stick drives the wheels
MODE_BUTTON = 0
DEADBAND = 0.08
W, H, SIDE = 1280, 720, 320
MODES = ["MANUAL", "AUTO", "POLICY"]


def keyboard():
    k = pygame.key.get_pressed()
    ax = lambda pos, neg: float(k[pos]) - float(k[neg])
    arm = dict(slew=ax(pygame.K_a, pygame.K_d), stick=ax(pygame.K_s, pygame.K_w),
               boom=ax(pygame.K_DOWN, pygame.K_UP), bucket=ax(pygame.K_LEFT, pygame.K_RIGHT))
    fwd, turn = ax(pygame.K_i, pygame.K_k), ax(pygame.K_j, pygame.K_l)
    return arm, fwd, turn


def joystick(js):
    v = {name: sign * js.get_axis(i) for i, (name, sign) in AXES.items() if i < js.get_numaxes()}
    v = {n: 0.0 if abs(x) < DEADBAND else x for n, x in v.items()}
    if DRIVE_SWITCH < js.get_numaxes() and js.get_axis(DRIVE_SWITCH) > 0:
        fwd, turn = -v.pop("boom", 0.0), -v.pop("bucket", 0.0)
        return v, fwd, turn
    return v, 0.0, 0.0


def to_action(arm, fwd, turn):
    a = np.zeros(6)
    a[:4] = [np.clip(arm.get(n, 0.0), -1, 1) * 0.6 for n in ARM]
    a[4], a[5] = np.clip(fwd - turn, -1, 1), np.clip(fwd + turn, -1, 1)
    return a


def calibrate():
    pygame.init()
    pygame.joystick.init()
    if not pygame.joystick.get_count():
        raise SystemExit("No joystick found. Put the transmitter in USB-joystick mode and plug it in.")
    js = pygame.joystick.Joystick(0)
    print(f"{js.get_name()}: {js.get_numaxes()} axes, {js.get_numbuttons()} buttons (Ctrl-C to stop)")
    while True:
        pygame.event.pump()
        print("\r" + " ".join(f"a{i}={js.get_axis(i):+.2f}" for i in range(js.get_numaxes()))
              + "  b=" + "".join(str(js.get_button(i)) for i in range(js.get_numbuttons())), end="")
        time.sleep(0.05)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--soil", default="loose")
    p.add_argument("--policy")
    p.add_argument("--calibrate", action="store_true")
    args = p.parse_args()
    if args.calibrate:
        return calibrate()

    env = ExcavatorEnv(soil=args.soil, episode_s=1e9)
    obs, _ = env.reset()
    pilot = Autopilot(env)
    policy = load_policy(args.policy) if args.policy else None
    pygame.init()
    pygame.joystick.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Excavator ground station")
    font = pygame.font.Font(None, 20)
    big = pygame.font.Font(None, 28)
    r = mujoco.Renderer(env.m, H, W - SIDE)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 4.0, 150, -25
    js = pygame.joystick.Joystick(0) if pygame.joystick.get_count() else None
    mode, estop, had_js, uploaded = 0, False, js is not None, 0
    dt = env.substeps * env.m.opt.timestep
    clock, t_sim0, t_wall0 = pygame.time.Clock(), env.d.time, time.time()

    def blit(img, xy, size=None):
        s = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        screen.blit(pygame.transform.smoothscale(s, size) if size else s, xy)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_TAB or \
                    ev.type == pygame.JOYBUTTONDOWN and ev.button == MODE_BUTTON:
                mode = (mode + 1) % (3 if policy else 2)
                pilot.reset()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                estop = not estop
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_BACKSPACE:
                obs, _ = env.reset()
                pilot.reset()
                t_sim0, t_wall0 = env.d.time, time.time()
            if ev.type == pygame.JOYDEVICEADDED and js is None:
                js, had_js = pygame.joystick.Joystick(ev.device_index), True
            if ev.type == pygame.JOYDEVICEREMOVED:
                js = None
            if ev.type == pygame.MOUSEMOTION and ev.buttons[0]:
                cam.azimuth -= ev.rel[0] * 0.3
                cam.elevation = np.clip(cam.elevation - ev.rel[1] * 0.3, -89, -2)
            if ev.type == pygame.MOUSEWHEEL:
                cam.distance = np.clip(cam.distance * (0.9 ** ev.y), 1.0, 15.0)

        link_lost = MODES[mode] == "MANUAL" and had_js and js is None
        # Step the sim to wall-clock time; if it falls behind, drop the backlog.
        target = t_sim0 + time.time() - t_wall0
        if target - env.d.time > 0.1:
            t_sim0, t_wall0 = env.d.time, time.time()
            target = env.d.time + dt
        info = env.info()
        while env.d.time < target:
            if estop or link_lost:
                a = np.zeros(6)
            elif MODES[mode] == "AUTO":
                a = pilot.act()
            elif MODES[mode] == "POLICY":
                pilot.update()
                a = policy(np.r_[obs, pilot.features()])
                a = np.r_[a, 0, 0] if len(a) == 4 else a
            else:
                arm, fwd, turn = keyboard()
                if js is not None:
                    ja, jf, jt = joystick(js)
                    arm = {n: arm[n] + ja.get(n, 0.0) for n in arm}
                    fwd, turn = fwd + jf, turn + jt
                a = to_action(arm, fwd, turn)
            obs, _, term, _, info = env.step(a)

        if env.terrain_version != uploaded:
            mujoco.mjr_uploadHField(env.m, r._mjr_context, env.m.geom_dataid[env.hid])
            uploaded = env.terrain_version
        cam.lookat[:] = env.d.qpos[:3] + [0, 0, 0.3]
        r.update_scene(env.d, cam)
        blit(r.render(), (0, 0))
        for i, name in enumerate(["mast", "bucket"]):
            r.update_scene(env.d, name)
            blit(r.render(), (W - SIDE, i * 240), (SIDE, 240))
        hud(screen, font, big, env, info, MODES[mode], estop, link_lost, js)
        pygame.display.flip()
        clock.tick(30)


def hud(screen, font, big, env, info, mode, estop, link_lost, js):
    x0, y = W - SIDE, 480
    pygame.draw.rect(screen, (18, 18, 22), (x0, y, SIDE, H - y))
    col = (230, 60, 60) if estop or link_lost else (255, 190, 40) if mode != "MANUAL" else (120, 200, 255)
    label = "E-STOP" if estop else "LINK LOST - HOLD" if link_lost else mode
    screen.blit(big.render(label, True, col), (x0 + 10, y + 6))
    screen.blit(font.render("TX " + (js.get_name()[:22] if js else "keyboard"), True, (150, 150, 150)), (x0 + 150, y + 10))
    lines = [f"bucket {info['fill_kg']:5.1f} kg   force {info['force_N']:6.0f} N",
             f"dumped {info['dumped_kg']:5.1f} kg   energy {info['energy_Wh']:5.2f} Wh",
             f"tilt {info['tilt_deg']:4.1f}°   tip margin {info['tip_margin']:+.2f} m"]
    for i, s in enumerate(lines):
        screen.blit(font.render(s, True, (220, 220, 220)), (x0 + 10, y + 36 + 18 * i))
    tau = np.abs(env.d.qfrc_applied[env.da]) / env.tau_max
    for i, (n, f) in enumerate(zip(ARM, tau)):
        yy = y + 96 + 26 * i
        screen.blit(font.render(f"{n:6s}", True, (180, 180, 180)), (x0 + 10, yy))
        pygame.draw.rect(screen, (50, 50, 55), (x0 + 80, yy + 3, 220, 12))
        c = (230, 60, 60) if f > 0.9 else (255, 190, 40) if f > 0.6 else (90, 200, 120)
        pygame.draw.rect(screen, c, (x0 + 80, yy + 3, int(220 * min(f, 1)), 12))


if __name__ == "__main__":
    main()
