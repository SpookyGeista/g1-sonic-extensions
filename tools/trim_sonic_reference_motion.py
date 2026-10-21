#!/usr/bin/env python3
"""Trim a SONIC reference motion and regenerate all velocity channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from merge_sonic_reference_motions import (
    FPS,
    calculate_velocities,
    load_motion,
    normalize_quat,
    save_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, required=True,
                        help="Inclusive 50 Hz frame index")
    args = parser.parse_args()

    motion = load_motion(args.input)
    frame_count = len(motion["joint_pos"])
    if not (0 <= args.start_frame <= args.end_frame < frame_count):
        parser.error(
            f"expected 0 <= start <= end < {frame_count}, got "
            f"{args.start_frame}..{args.end_frame}"
        )
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")

    selection = slice(args.start_frame, args.end_frame + 1)
    joint_pos = motion["joint_pos"][selection].copy()
    body_pos = motion["body_pos"][selection].copy()
    body_quat = normalize_quat(motion["body_quat"][selection].copy())
    joint_vel, body_lin_vel, body_ang_vel = calculate_velocities(
        joint_pos, body_pos, body_quat
    )

    arrays = (joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel)
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Trimmed motion contains NaN or infinity")

    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "joint_pos.csv", joint_pos,
             [f"joint_{i}" for i in range(29)])
    save_csv(args.output / "joint_vel.csv", joint_vel,
             [f"joint_vel_{i}" for i in range(29)])
    save_csv(args.output / "body_pos.csv", body_pos,
             [f"body_{i}_{axis}" for i in range(14) for axis in "xyz"])
    save_csv(args.output / "body_quat.csv", body_quat,
             [f"body_{i}_{axis}" for i in range(14) for axis in "wxyz"])
    save_csv(args.output / "body_lin_vel.csv", body_lin_vel,
             [f"body_{i}_vel_{axis}" for i in range(14) for axis in "xyz"])
    save_csv(args.output / "body_ang_vel.csv", body_ang_vel,
             [f"body_{i}_angvel_{axis}" for i in range(14) for axis in "xyz"])

    (args.output / "metadata.txt").write_text(
        f"Metadata for: {args.output.name}\n{'=' * 30}\n\n"
        "Body part indexes:\n[ 0  4 10 18  5 11 19  9 16 22 28 17 23 29]\n\n"
        f"Total timesteps: {len(joint_pos)}\n",
        encoding="utf-8",
    )
    metadata = {
        "motion_name": args.output.name,
        "source_motion": args.input.as_posix(),
        "source_start_frame": args.start_frame,
        "source_end_frame_inclusive": args.end_frame,
        "fps": FPS,
        "frames": len(joint_pos),
        "duration_seconds": (len(joint_pos) - 1) / FPS,
        "velocity_channels": "regenerated after trimming",
    }
    (args.output / "trim_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
