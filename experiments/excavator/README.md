# Remote-operated micro-excavator

`src/draft/robots/excavator/` is a ~162 kg (172 kg loaded) electric micro-excavator that is remote-operated, so it has no cab or seat. It runs on a
skid-steer wheeled undercarriage (battery in the chassis) with a slewing deck that carries the compute, the radio, an FPV mast
camera (`cam_mast`), a counterweight and a boom/stick/bucket arm. A second camera (`cam_bucket`) sits under the stick. The reach at ground level is 1.65 m.

```bash
pip install pygame gymnasium stable-baselines3 torch tensorboard   # on top of the draft install
python scripts/generate_robot.py --robot excavator --output-dir generated/excavator --fixed-output-dir
python experiments/excavator/check_reach.py      # holding torque + tip-over across slew/arm poses
python experiments/excavator/sim_and_render.py   # dig cycle + drive, writes render_*.png
mjpython experiments/excavator/sim_and_render.py --view   # same cycle, live 3D viewer
python scripts/robot_tuner.py                    # browser editor
python experiments/excavator/autopilot.py        # autonomous dig/dump benchmark per soil
python experiments/excavator/ground_station.py   # remote operation: FPV cams, HUD, MANUAL/AUTO/POLICY
python experiments/excavator/train.py --out runs/arm_v1   # demos → behaviour cloning → PPO
```

## Operate, automate, learn

`env.py` puts the machine on an 8 m diggable heightfield. Every driver uses one action,
`[slew, boom, stick, bucket, left, right]` in [-1, 1] at 50 Hz: the operator's sticks, the autopilot,
and a trained policy alike. A zero command holds position, so it acts as a brake. Soil resists the teeth by the fundamental
earthmoving equation `F = w(γgd²Nγ + c·d·Nc)`. Scooped cells become bucket fill, the fill weighs on
the bucket and spills when the bucket tips, and spoil slumps to a 35° angle of repose. Soil presets: loose, firm, clay.

- **`autopilot.py`**: arm inverse kinematics plus a swing → cut → curl → swing → dump state machine. The cut is
  force-limited: the target only advances while the arm keeps up, and the cut gets shallower if the bucket stalls.
- **`ground_station.py`**: chase view, mast FPV, bucket camera, and a torque/tip-margin readout. Tab cycles
  MANUAL / AUTO / POLICY and Space is the e-stop. If the link drops in MANUAL, the machine holds position. For a drone transmitter, set it to
  USB-joystick mode, run `--calibrate`, then edit `AXES` / `DRIVE_SWITCH`.
- **`train.py`**: self-operation is a planner plus a learned controller. The autopilot's planner
  (current phase, next teeth waypoint, advanced from the measured pose) is part of the policy's input, and
  the policy learns the joint commands, including cuts into soil it can't see. Pipeline: noisy autopilot demos
  (DART) → behaviour cloning → PPO across 12 CPU processes, at about 2.5k steps/s. Without the planner input,
  cloning scored 0 kg; with it, the clone matches the autopilot (21.8 kg per 30 s episode). Stage 1 is arm only;
  `--full` adds the drive. Run `--eval runs/x/model.zip` to score a policy, or
  `ground_station.py --policy runs/x/model.zip` to watch it.

Learned vs scripted, on 10 identical randomized 60 s scenes. The policy is `runs/arm_v2/model_1.5M.zip`: behaviour cloning,
then 1.5M PPO steps on 30 s episodes, then 1.5M on 60 s episodes. The 30 s episodes only had room for one
bucket, which capped the score; switching to 60 s took it from 57 to 65 kg. Training stopped at 2.0M steps of the second stage, at a plateau
(65.1 kg); `model.zip` is that checkpoint. `runs/` holds models and TensorBoard logs and isn't gitignored.
The randomized scenes vary the dig and dump bearings, the soil and the start pose, so the autopilot does fewer cycles there than in
the fixed benchmark above:

| controller | dumped kg/min | spilled | energy | min tip margin | tipped |
|---|---|---|---|---|---|
| autopilot | 42.1 ± 4.1 | 0.2 kg | 1.09 Wh | +0.07 m | 0/10 |
| learned | 63.1 ± 6.0 | 0.5 kg | 1.26 Wh | +0.07 m | 0/10 |

The learned controller moves 50% more soil, using 20% less energy per kg. It levels off at about 65 kg/min
(three buckets) because the planner's waypoint pacing becomes the limit. The next stage is to learn the waypoints too, or add the drive.

Autopilot, 5 cycles per soil (in a 1/4 m² dump zone):

| soil | kg/cycle | kg/h | peak teeth force | peak τ/limit (slew boom stick bucket) | min tip margin |
|---|---|---|---|---|---|
| loose | 18.9 | 3070 | 0.3 kN | 0.56 0.60 0.87 0.85 | 0.09 m |
| firm | 21.5 | 3390 | 1.0 kN | 0.56 0.71 1.00 1.00 | 0.08 m |
| clay | 22.8 | 3600 | 3.3 kN | 0.57 0.72 1.00 1.00 | 0.07 m |

Findings: in firm soil and clay the stick and bucket hit their torque limits; the bucket is saturated 5% of the time in clay. The boom never uses more than 72%
of its limit. The next sizing step is a larger bucket drive (e.g. eRob90H-100), keeping the boom as it is.
Mechanical energy is about 0.5 Wh per cycle. The simulation also found that the deck hit the wheels when slewing; `deck_base`
now raises it clear.

| joint | anchor (catalogue) | peak N·m | static demand, 10 kg payload at full reach |
|---|---|---|---|
| boom | eRob142H-100 | 841 | 363 (2.3×) |
| stick | eRob110H-100 | 369 | 130 (2.8×) |
| bucket | eRob80H-100 | 143 | 28 |
| slew | eRob90H-100 | 191 | — |
| wheels ×4 | AK80-64 | 120 | ~0.9 m/s top speed |

The design passes every pose in the stability check. The CoM stays ≥0.12 m inside the wheel footprint, including at full reach
over the side, and `counterweight_mass` is the tuning knob.

**What the numbers do not cover.** The arm structure is *stated* (an equivalent steel box-section radius and
density) rather than derived from trends, because Draft's humanoid and quadruped populations don't cover
excavator booms. The 2–3× headroom is the margin for digging reaction and breakout force, but neither is modelled. As
Draft intends, the MJCF has no `<actuator>` block, so the controller belongs to whoever uses the model.

**Teleop mapping** (drone-style transmitter, velocity targets into joint PD loops like `sim_and_render.py`):
left stick X/Y = slew / boom, right stick X/Y = bucket / stick (ISO excavator pattern). Use the left and right
throttles or triggers for the skid-steer wheel speeds, and a switch for drive mode versus dig mode.
