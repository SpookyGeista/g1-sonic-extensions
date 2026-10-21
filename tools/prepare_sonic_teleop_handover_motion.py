#!/usr/bin/env python3
"""Append a smooth return to frame zero to a SONIC reference motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from merge_sonic_reference_motions import (
    FPS,
    bridge,
    calculate_velocities,
    load_motion,
    normalize_quat,
    save_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--return-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    args = parser.parse_args()

    return_frames = round(args.return_seconds * FPS)
    if return_frames < 1:
        parser.error("--return-seconds must produce at least one frame")
    hold_frames = round(args.hold_seconds * FPS)
    if hold_frames < 1:
        parser.error("--hold-seconds must produce at least one frame")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")

    motion = load_motion(args.input)
    joint_pos = np.concatenate(
        [
            motion["joint_pos"],
            bridge(motion["joint_pos"][-1], motion["joint_pos"][0], return_frames),
            np.repeat(motion["joint_pos"][:1], hold_frames, axis=0),
        ]
    )
    body_pos = np.concatenate(
        [
            motion["body_pos"],
            bridge(motion["body_pos"][-1], motion["body_pos"][0], return_frames),
            np.repeat(motion["body_pos"][:1], hold_frames, axis=0),
        ]
    )
    body_quat = normalize_quat(
        np.concatenate(
            [
                motion["body_quat"],
                bridge(
                    motion["body_quat"][-1],
                    motion["body_quat"][0],
                    return_frames,
                    quaternion=True,
                ),
                np.repeat(motion["body_quat"][:1], hold_frames, axis=0),
            ]
        )
    )
    joint_vel, body_lin_vel, body_ang_vel = calculate_velocities(
        joint_pos, body_pos, body_quat
    )

    arrays = [joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel]
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Prepared motion contains NaN or infinity")
    quat_error = float(np.max(np.abs(np.linalg.norm(body_quat, axis=-1) - 1.0)))
    if quat_error > 1e-6:
        raise ValueError(f"Quaternion norm error: {quat_error}")

    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "joint_pos.csv", joint_pos, [f"joint_{i}" for i in range(29)])
    save_csv(args.output / "joint_vel.csv", joint_vel, [f"joint_vel_{i}" for i in range(29)])
    save_csv(
        args.output / "body_pos.csv",
        body_pos,
        [f"body_{i}_{axis}" for i in range(14) for axis in "xyz"],
    )
    save_csv(
        args.output / "body_quat.csv",
        body_quat,
        [f"body_{i}_{axis}" for i in range(14) for axis in "wxyz"],
    )
    save_csv(
        args.output / "body_lin_vel.csv",
        body_lin_vel,
        [f"body_{i}_vel_{axis}" for i in range(14) for axis in "xyz"],
    )
    save_csv(
        args.output / "body_ang_vel.csv",
        body_ang_vel,
        [f"body_{i}_angvel_{axis}" for i in range(14) for axis in "xyz"],
    )

    name = args.output.name
    (args.output / "metadata.txt").write_text(
        f"Metadata for: {name}\n{'=' * 30}\n\n"
        "Body part indexes:\n[ 0  4 10 18  5 11 19  9 16 22 28 17 23 29]\n\n"
        f"Total timesteps: {len(joint_pos)}\n",
        encoding="utf-8",
    )
    metadata = {
        "motion_name": name,
        "source_motion": args.input.as_posix(),
        "fps": FPS,
        "source_frames": len(motion["joint_pos"]),
        "return_frames": return_frames,
        "return_seconds": return_frames / FPS,
        "hold_frames": hold_frames,
        "hold_seconds": hold_frames / FPS,
        "frames": len(joint_pos),
        "duration_seconds": (len(joint_pos) - 1) / FPS,
        "final_frame_equals_source_start": {
            "joint_pos": bool(np.array_equal(joint_pos[-1], motion["joint_pos"][0])),
            "body_pos": bool(np.array_equal(body_pos[-1], motion["body_pos"][0])),
            "body_quat_rotation": bool(
                np.allclose(
                    np.abs(np.sum(body_quat[-1] * motion["body_quat"][0], axis=-1)),
                    1.0,
                    atol=1e-9,
                )
            ),
        },
        "max_quaternion_norm_error": quat_error,
        "max_abs_joint_velocity_rad_s": float(np.max(np.abs(joint_vel))),
    }
    (args.output / "handover_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
