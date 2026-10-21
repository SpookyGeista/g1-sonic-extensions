# SONIC tool workflows

Run these commands from the patched upstream checkout after copying `tools/` there. Paths beginning with `/path/to/` are inputs to supply. Create a fresh output directory for each transformation.

For live switching, use the separate [recorded-motion / teleoperation guide](MOTION_CAPTURE_TELEOP.md).

## Reference format and dependencies

A SONIC reference uses 50 Hz samples: 29 joint positions in radians, 14 body positions in metres, and `wxyz` body quaternions. A complete output includes `joint_pos.csv`, `joint_vel.csv`, `body_pos.csv`, `body_quat.csv`, `body_lin_vel.csv`, `body_ang_vel.csv`, and `metadata.txt`.

| Work | Environment |
|---|---|
| Trim, merge, hold, and handover preparation | Python with NumPy. |
| BONES/Kimodo conversion and kinematic preview | MuJoCo, NumPy, and the configured G1 MJCF/meshes; preview video also uses OpenCV. |
| Route plots | Matplotlib and NumPy. |
| Dataset download | Hugging Face Hub and network access. |
| Kimodo generation | Configured Kimodo checkout, model, and its environment. |
| Isaac re-rendering | Configured Isaac environment, retained scene files, SciPy, and FFmpeg. |
| Tracking diagnostics | Isaac Teleop runtime and the connected tracking source. |

The standalone `requirements-offline.txt` only covers the documented small test subset. Check each tool's `--help` after its dependencies are installed.

## 1. Download and convert BONES motion capture

```bash
python tools/download_bones_seed_selected.py --output-root data/bones_selected
python tools/convert_bones_to_sonic_reference.py \
  --manifest data/bones_selected/manifest.csv \
  --output-root data/sonic_reference \
  --mjcf gear_sonic_deploy/g1/g1_29dof.xml
```

The downloader selects the clips defined in its source and writes `manifest.csv`. The converter turns BONES G1 CSVs into SONIC reference directories and saves conversion metadata. Dataset licensing and access are managed at the dataset source.

## 2. Trim, combine, and prepare a handover

```bash
python tools/trim_sonic_reference_motion.py \
  --input /path/to/reference --output data/trimmed \
  --start-frame 0 --end-frame 99
python tools/merge_sonic_reference_motions.py \
  --first /path/to/first --second /path/to/second \
  --output data/merged --bridge-seconds 0.5
python tools/build_sonic_module_sequence.py \
  --entry /path/to/entry --loop /path/to/loop --loop-count 3 \
  --exit /path/to/exit --output data/sequence --bridge-seconds 0.5
python tools/prepare_sonic_teleop_handover_motion.py \
  --input data/sequence --output data/handover_playlist/01_ready \
  --return-seconds 2 --hold-seconds 1
```

The trim end frame is inclusive. Merge aligns the second clip and inserts a bridge; the sequence tool chains entry, repeated loop, and exit. These tools regenerate velocity channels. Handover preparation returns to the source's first pose, then holds it.

For a longer stationary ending:

```bash
python tools/extend_sonic_reference_hold.py \
  --input data/sequence --output data/sequence_hold --hold-seconds 10
```

## 3. Preview a reference and plot its route

```bash
python tools/preview_sonic_reference.py \
  --motion data/sequence --interactive
python tools/plot_bones_crawl_route.py \
  --motion data/sequence --output data/sequence_route.png \
  --title "Reference route"
```

The preview places recorded poses kinematically. For comparison with an actual recorded Isaac trajectory:

```bash
python tools/plot_actual_bones_route.py \
  --motion data/sequence --trial /path/to/recorded_trial \
  --output data/measured_route.png
```

The measured-route plot needs a compatible completed trial with controller and physics records.

## 4. Kimodo conversion

```bash
python tools/convert_kimodo_to_sonic_reference.py \
  --input /path/to/g1_qpos.csv --output data/kimodo_reference \
  --source-fps 30 --kimodo-root /path/to/kimodo \
  --mjcf gear_sonic_deploy/g1/g1_29dof.xml
```

Set `--source-fps` to the recording's actual sample rate. The converter defaults to rejecting joint-limit violations. `generate_kimodo_demo_clips.py --output data/kimodo_plan --plan-only` writes the selected generation plan; actual generation requires the external model environment.

## 5. Recorded-video exports

```bash
python tools/render_isaac_recorded_videos.py \
  --scene-dir /path/to/scene --trial /path/to/recorded_trial \
  --output data/video_export --edit /path/to/edit.json
python tools/validate_isaac_recorded_videos.py \
  --export-dir data/video_export --output data/video_validation.json \
  --ffmpeg /path/to/ffmpeg --ffprobe /path/to/ffprobe
```

The renderer needs retained scene/runtime files and uses a local FFmpeg path in its source; adapt it before use. Validation checks the encoded files and edit. These operations use existing recorded trajectories.

## Tool index

| Script in `tools/` | Inputs and result |
|---|---|
| `download_bones_seed_selected.py` | Selected remote BONES clips → local CSVs and download manifest. |
| `convert_bones_to_sonic_reference.py` | BONES CSV or manifest + MJCF → 50 Hz references. |
| `convert_kimodo_to_sonic_reference.py` | Kimodo qpos + sample rate/model paths → references. |
| `trim_sonic_reference_motion.py` | Reference + frame interval → trimmed reference. |
| `merge_sonic_reference_motions.py` | Two references + bridge duration → merged clip. |
| `build_sonic_module_sequence.py` | Entry, loop, repetition count, exit → one sequence. |
| `prepare_sonic_teleop_handover_motion.py` | Reference + return/hold durations → handover-ready clip. |
| `extend_sonic_reference_hold.py` | Reference + hold duration → extended final pose. |
| `build_bones_crouch_pair.py` | BONES dataset directory → explicit stand/crouch sequences. |
| `build_bones_continuous.py` | `--spec` sequence JSON + `--output` → trimmed/overlapped chain; sources must be inside the checkout. |
| `build_bones_crawl_extension.py` | `--spec` + `--output` → chain with ground-motion extension. |
| `build_bones_walk_video_pack.py` | `--spec`, `--output`, optional `--sequence` → walking modules and custom chain. |
| `trim_bones_motion_pauses.py` | `--spec` + `--output` → shortened reviewed idle intervals. |
| `build_bones_wall_contact_layout.py` | Scene `--source`, `--output`, `--left-x`, `--upper-y` → fitted wall layout. |
| `preview_sonic_reference.py` | Reference → interactive kinematic preview, video, contact sheet, or report. |
| `plot_bones_crawl_route.py` | Reference → route and height plot. |
| `plot_actual_bones_route.py` | Reference + recorded trial → measured/reference comparison. |
| `plot_bones_apartment.py` | `--layout`, `--motion`, `--trial`, `--output` → building/trajectory plan. |
| `render_bones_wall_contact_poses.py` | `--scene-dir`, `--trial`, `--output` → recorded contact-pose views. |
| `render_isaac_recorded_videos.py` | Scene, recorded trial, and edit JSON → camera videos. |
| `validate_isaac_recorded_videos.py` | Video export + FFmpeg/FFprobe → integrity report. |
| `generate_kimodo_demo_clips.py` | Clip selection, seed, model → generation plan or candidates. |
| `debug_isaac_teleop_body.py` | Tracking source → raw body-sample diagnostics; `--jsonl` saves samples. |

Spec-driven tools need the JSON schema used by their source; historical scene/sequence specifications are not all included in this export. `tests/` and `test_*.py` files are checks, rather than additional launchers.
