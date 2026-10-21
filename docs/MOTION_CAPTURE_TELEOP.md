# Switching between recorded motion and live teleoperation

This workflow plays a recorded motion-capture reference, pauses it for live full-body POSE teleoperation, and resumes the same reference at its saved frame. The switch is handled by the deployment terminal's **Enter** key.

## Before starting

Apply the patch and copy the tools into the upstream checkout as described in the [README](../README.md#apply-the-integration-changes). Install the upstream simulator, teleoperation environment, policy checkpoint, and deployment dependencies. Commands below run from that patched checkout and use the existing `.venv_sim` and `.venv_teleop` environments.

The soft switch requires `--input-type zmq` and `--soft-teleop-handover`. It uses full-body SMPL packets in encoder mode 2. The examples explicitly select the simulation interface with the final `sim` argument.

## 1. Prepare a motion and playlist

Start with a converted 50 Hz SONIC reference. Use a clip whose first pose is a suitable neutral standing pose:

```bash
source .venv_sim/bin/activate
python tools/prepare_sonic_teleop_handover_motion.py \
  --input /path/to/converted_motion \
  --output data/handover_playlist/01_ready \
  --return-seconds 2.0 --hold-seconds 1.0
```

This appends a two-second return to the clip's exact first root/joint pose and a one-second hold. It writes the six CSV channels, `metadata.txt`, and `handover_metadata.json`. Use a new output directory. `data/handover_playlist` is the playlist; `01_ready` is its motion subdirectory. Motion data is prepared locally, rather than included in this repository.

## 2. Start the simulator in terminal 1

From the patched upstream checkout:

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

## 3. Start the motion-capture streamer in terminal 2

From the same checkout:

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager --input-source isaac-teleop --vis_smpl --vis_vr3pt
```

Connect the configured Quest/PICO tracking source. While standing neutrally, press **A+B+X+Y** to leave OFF mode and calibrate. Then press **A+X** to change from PLANNER to **POSE**. Check the streamer's current mode before switching the deployment. Keep it in POSE throughout this workflow.

The streamer publishes `pose` packets on port 5556 by default. The controller-button mode selection and the deployment's Enter switch are separate controls.

## 4. Start SONIC deployment in terminal 3

From the same checkout:

```bash
cd gear_sonic_deploy
bash deploy.sh \
  --input-type zmq --zmq-host localhost \
  --motion-data ../data/handover_playlist \
  --soft-teleop-handover \
  --teleop-blend-seconds 3.0 \
  --teleop-max-joint-speed 1.0 \
  sim
```

The script builds the deployment binary before launching it. Use the checkpoint and observation-configuration options from the upstream installation when their files differ from the script defaults.

## 5. Play, take over, and resume

Keep keyboard focus in **terminal 3**:

| Input | Action |
|---|---|
| `]` | Enable SONIC control. |
| `N` / `P` | Select the next / previous loaded motion. |
| `T` | Start or continue reference playback. |
| `Enter` during reference playback | Save the current motion/frame and hold its pose while waiting for a fresh SMPL packet; then blend into live full-body teleoperation. |
| `Enter` during teleoperation | Restore the saved motion/frame and encoder mode, then blend back into reference playback. |
| `O` | Stop the controller. |

For a first trial, play the prepared clip through its neutral ending and then press Enter. The final reference frame is held until the switch. A switch during an earlier frame resumes at that earlier saved frame.

Expected deployment messages include:

```text
FULL-BODY TELEOP: WAITING FOR FRESH SMPL DATA
[FullBodyTeleop] Motion paused; holding its final G1 pose
[FullBodyTeleop] Fresh SMPL stream active in encoder mode 2; starting soft handover
[TeleopHandover] Soft handover complete
```

On the return switch:

```text
FULL-BODY TELEOP: DISABLED
Smoothly resuming the paused motion
```

## Transition settings and troubleshooting

| Setting or symptom | Meaning / action |
|---|---|
| `--teleop-blend-seconds 3.0` | Nominal quintic interpolation time for motor targets. |
| `--teleop-max-joint-speed 1.0` | Per-joint target slew limit in rad/s; it can make settling take longer than the nominal blend time. |
| Waiting for fresh SMPL data | Check the tracking source, POSE mode, publisher, host, and port. Old buffered packets are cleared when switching. |
| Enter has no effect | Focus terminal 3 and verify both soft-handover flags were supplied. |
| Unknown soft-handover option | Verify the patch is applied to the recorded base revision and the C++ deployment was rebuilt. |
| Large pose mismatch | Return to a pose close to the held reference before switching. Blended commands do not establish balance for arbitrary pose changes. |

Calibration happens in the streamer. Enter requests the handover and heading reinitialization; it does not run a new three-point calibration. This guide describes the patched source behavior, not a new live-headset or hardware validation.
