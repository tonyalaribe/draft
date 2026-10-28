#!/usr/bin/env python3
"""Train the excavator's learned controller: autopilot demos → behaviour cloning → PPO.

Self-operation is a planner plus a learned controller. The autopilot's planner (phase and
next teeth waypoint, advanced from the measured pose) is part of the observation; the policy
learns the joint commands that get there, including force-limited cuts in soil it cannot see.
Curriculum stage 1 (default): arm only, wheels braked, randomised dig/dump bearings, soil
and start pose. Reward = kg in the dump zone + 0.3·kg scooped − spill − tipping, plus
potential-based progress toward the waypoint and a bonus per waypoint reached.

    python experiments/excavator/train.py --demos 48 --bc-epochs 30 --steps 3e6 --out runs/arm_v1
    python experiments/excavator/train.py --eval runs/arm_v1/model.zip
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from autopilot import Autopilot
from env import SOILS, ExcavatorEnv


class DigTask(gym.Env):
    def __init__(self, arm_only=True, episode_s=30.0, seed=0, norm=None):
        self.env = ExcavatorEnv(episode_s=episode_s, seed=seed)
        self.planner = Autopilot(self.env)
        self.arm_only, self.rng = arm_only, np.random.default_rng(seed)
        n = 4 if arm_only else 6
        self.action_space = gym.spaces.Box(-1, 1, (n,), np.float32)
        self.planner.update()
        dim = len(self.observe())
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32)
        self.norm = norm or (np.zeros(dim), np.ones(dim))

    def randomise(self):
        e, rng = self.env, self.rng
        th, r = rng.uniform(-0.5, 0.5), rng.uniform(0.9, 1.05)
        e.dig_site = r * np.array([np.cos(th), np.sin(th)])
        side = rng.choice([-1, 1])
        th2, r2 = side * rng.uniform(1.2, 1.9), rng.uniform(1.1, 1.3)
        e.dump_site = r2 * np.array([np.cos(th2), np.sin(th2)])
        base = SOILS[rng.choice(list(SOILS))]
        e.soil = dict(rho=base["rho"] * rng.uniform(0.9, 1.1), c=base["c"] * rng.uniform(0.7, 1.3))
        return [rng.uniform(-0.3, 0.3), rng.uniform(0.6, 1.0), rng.uniform(1.4, 2.0), rng.uniform(1.6, 2.1)]

    def observe(self):
        return np.r_[self.env.obs(), self.planner.features()]

    def progress(self, goal):
        e = np.abs(goal - self.planner.pose())
        return -(e[0] + e[1] + 0.3 * e[2] + 0.5 * min(e[3], 2 * np.pi - e[3]))

    def normalise(self, o):
        return ((o - self.norm[0]) / self.norm[1]).clip(-10, 10).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        _, info = self.env.reset(qpos=self.randomise())
        self.planner.reset()
        self.planner.update()
        self.prev = self.env.info()
        return self.normalise(self.observe()), info

    def step(self, a):
        a = np.r_[a, 0, 0] if self.arm_only else a
        goal = self.planner.goal()
        before = self.progress(goal)
        _, _, term, trunc, info = self.env.step(a)
        shaping = 5.0 * (self.progress(goal) - before) + 1.0 * self.planner.update()
        p = self.prev
        reward = ((info["dumped_kg"] - p["dumped_kg"]) + 0.3 * (info["dug_kg"] - p["dug_kg"])
                  - 0.5 * max(0.0, info["spilled_kg"] - p["spilled_kg"]) - 1e-3 * float(np.square(a).sum())
                  - 50.0 * term) / 10.0 + shaping
        self.prev = info
        return self.normalise(self.observe()), reward, term, trunc, info


def collect(n_episodes, out, arm_only=True, noise=0.3):
    """Autopilot demos; actions executed with noise (DART) so the clone sees recoveries, labels clean."""
    task, O, A, kg = DigTask(arm_only, seed=123), [], [], []
    for ep in range(n_episodes):
        task.reset()
        pilot = task.planner
        pilot.target = None
        for _ in range(task.env.max_steps):
            if pilot.target is None:
                pilot.target = pilot.pose()
            a = pilot.act(update=False)
            O.append(task.observe()), A.append(a[:4] if arm_only else a)
            _, _, term, trunc, info = task.env.step(a + task.rng.normal(0, noise, a.shape) * (ep % 4 != 0))
            pilot.update()
            if term or trunc:
                break
        kg.append(info["dumped_kg"])
    O, A = np.array(O, np.float32), np.array(A, np.float32)
    np.savez_compressed(out, obs=O, act=A)
    print(f"demos: {n_episodes} episodes, {len(O)} steps, autopilot {np.mean(kg):.1f} ± {np.std(kg):.1f} kg/episode")
    return O, A


def behaviour_clone(model, O, A, norm, epochs, batch=1024):
    pol = model.policy
    X = torch.as_tensor((O - norm[0]) / norm[1], dtype=torch.float32).clamp(-10, 10).to(pol.device)
    Y = torch.as_tensor(A, dtype=torch.float32).to(pol.device)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm, total = torch.randperm(len(X)), 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            mean = pol.action_net(pol.mlp_extractor.forward_actor(pol.extract_features(X[idx])))
            loss = torch.nn.functional.mse_loss(mean, Y[idx])
            opt.zero_grad(), loss.backward(), opt.step()
            total += loss.item() * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"bc epoch {ep:3d}  mse {total / len(X):.4f}")
    with torch.no_grad():
        pol.log_std.fill_(-1.5)  # start PPO close to the cloned behaviour


def evaluate(path, episodes=6, seed=1000, episode_s=60.0):
    pol = load_policy(path)
    arm_only = json.loads((Path(path).parent / "config.json").read_text())["arm_only"]
    kg = []
    for i in range(episodes):
        task = DigTask(arm_only, seed=seed + i, episode_s=episode_s)
        task.reset()
        for _ in range(task.env.max_steps):
            a = pol(task.observe())
            _, _, term, trunc, info = task.step(a)
            if term or trunc:
                break
        kg.append(info["dumped_kg"])
    return np.array(kg)


def load_policy(path):
    """[env obs, planner features] → action, with the training normalisation applied."""
    from stable_baselines3 import PPO
    path = Path(path)
    model = PPO.load(path, device="cpu")
    mu, sd = (np.load(path.parent / "norm.npz")[k] for k in ("mu", "sd"))
    act = lambda o: model.predict(((o - mu) / sd).clip(-10, 10).astype(np.float32), deterministic=True)[0]
    return act


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/arm_v1")
    p.add_argument("--demos", type=int, default=48)
    p.add_argument("--bc-epochs", type=int, default=30)
    p.add_argument("--steps", type=float, default=3e6)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--full", action="store_true", help="include the drive (stage 2)")
    p.add_argument("--eval")
    p.add_argument("--episode", type=float, default=60.0)
    p.add_argument("--init", help="resume PPO from a run directory (skips demos and BC)")
    args = p.parse_args()
    if args.eval:
        kg = evaluate(args.eval)
        return print(f"policy: {kg.mean():.1f} ± {kg.std():.1f} kg/episode ({kg.round(1)})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arm_only = not args.full
    (out / "config.json").write_text(json.dumps(dict(arm_only=arm_only, **vars(args))))
    if args.init:
        src = Path(args.init)
        nz = np.load(src / "norm.npz")
        norm = (nz["mu"], nz["sd"])
    else:
        O, A = collect(args.demos, out / "demos.npz", arm_only)
        norm = (O.mean(0), O.std(0) + 1e-3)
    np.savez(out / "norm.npz", mu=norm[0], sd=norm[1])

    venv = SubprocVecEnv([lambda i=i: DigTask(arm_only, args.episode, seed=i, norm=norm) for i in range(args.envs)])
    if args.init:
        model = PPO.load(src / "model.zip", env=venv, device="cpu", tensorboard_log=str(out / "tb"))
    else:
        model = PPO("MlpPolicy", venv, n_steps=512, batch_size=4096, n_epochs=5, learning_rate=1e-4,
                    gamma=0.998, gae_lambda=0.95, clip_range=0.15, ent_coef=0.0, device="cpu",
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                    tensorboard_log=str(out / "tb"), verbose=0)
        behaviour_clone(model, O, A, norm, args.bc_epochs)
    model.save(out / "start.zip")
    kg = evaluate(out / "start.zip")
    print(f"start: {kg.mean():.1f} ± {kg.std():.1f} kg per {args.episode:.0f} s episode")

    chunk, done, t0 = min(500_000, int(args.steps)), 0, time.time()
    while done < args.steps:
        model.learn(chunk, reset_num_timesteps=False)
        done += chunk
        model.save(out / "model.zip")
        kg = evaluate(out / "model.zip")
        print(f"{done / 1e6:5.1f}M steps  {done / (time.time() - t0):6.0f} steps/s  "
              f"policy {kg.mean():5.1f} ± {kg.std():4.1f} kg/episode", flush=True)


if __name__ == "__main__":
    main()
