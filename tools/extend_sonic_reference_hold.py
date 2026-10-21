#!/usr/bin/env python3
"""Append a stationary final-pose hold to a SONIC reference motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MOTION_FILES = {
    "joint_pos.csv": 29,
    "joint_vel.csv": 29,
    "body_pos.csv": 42,
    "body_quat.csv": 56,
    "body_lin_vel.csv": 42,
    "body_ang_vel.csv": 42,
}
STATIONARY_FILES = {"joint_vel.csv", "body_lin_vel.csv", "body_ang_vel.csv"}


def load_csv(path: Path, width: int) -> tuple[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    values = np.atleast_2d(values)
    if values.shape[1] != width:
        raise ValueError(f"{path} has shape {values.shape}; expected (*, {width})")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains NaN or infinity")
    return header, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hold-seconds", type=float, default=120.0)
    parser.add_argument("--fps", type=float, default=50.0)
    args = parser.parse_args()

    if not np.isfinite(args.hold_seconds) or args.hold_seconds <= 0.0:
        parser.error("--hold-seconds must be finite and greater than zero")
    if not np.isfinite(args.fps) or args.fps <= 0.0:
        parser.error("--fps must be finite and greater than zero")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    hold_frames = round(args.hold_seconds * args.fps)
    loaded: dict[str, tuple[str, np.ndarray]] = {}
    original_frames: int | None = None
    for filename, width in MOTION_FILES.items():
        header, values = load_csv(args.input / filename, width)
        if original_frames is None:
            original_frames = len(values)
        elif len(values) != original_frames:
            raise ValueError(f"Frame-count mismatch in {filename}")
        loaded[filename] = (header, values)

    assert original_frames is not None
    args.output.mkdir(parents=True)
    for filename, (header, values) in loaded.items():
        if filename in STATIONARY_FILES:
            hold = np.zeros((hold_frames, values.shape[1]), dtype=np.float64)
        else:
            hold = np.repeat(values[-1:, :], hold_frames, axis=0)
        extended = np.concatenate([values, hold], axis=0)
        np.savetxt(
            args.output / filename,
            extended,
            delimiter=",",
            header=header,
            comments="",
            fmt="%.6f",
        )

    body_quat = np.concatenate(
        [loaded["body_quat.csv"][1], np.repeat(loaded["body_quat.csv"][1][-1:], hold_frames, axis=0)]
    ).reshape(-1, 14, 4)
    quaternion_error = float(np.max(np.abs(np.linalg.norm(body_quat, axis=-1) - 1.0)))
    total_frames = original_frames + hold_frames
    total_duration = (total_frames - 1) / args.fps

    source_metadata = args.input / "metadata.txt"
    body_indexes = "[ 0  4 10 18  5 11 19  9 16 22 28 17 23 29]"
    if source_metadata.exists():
        text = source_metadata.read_text(encoding="utf-8")
        if "Body part indexes:" in text:
            body_indexes = text.split("Body part indexes:", 1)[1].strip().splitlines()[0]
    (args.output / "metadata.txt").write_text(
        f"Metadata for: {args.output.name}\n{'=' * 30}\n\n"
        f"Body part indexes:\n{body_indexes}\n\n"
        f"Total timesteps: {total_frames}\n",
        encoding="utf-8",
    )

    metadata = {
        "motion_name": args.output.name,
        "source_motion": args.input.as_posix(),
        "fps": args.fps,
        "original_frames": original_frames,
        "hold_frames": hold_frames,
        "hold_seconds": hold_frames / args.fps,
        "total_frames": total_frames,
        "total_duration_seconds": total_duration,
        "hold_behavior": "final joint/body pose repeated; joint/body linear/body angular velocities set to zero",
        "max_quaternion_norm_error": quaternion_error,
    }
    (args.output / "hold_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
