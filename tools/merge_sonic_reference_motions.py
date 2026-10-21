#!/usr/bin/env python3
"""Merge two SONIC deployment reference motions into one continuous clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FPS = 50.0


def normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    result = q.copy()
    result[..., 1:] *= -1.0
    return result


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def yaw_quat(q: np.ndarray) -> np.ndarray:
    """Return the normalized yaw-only component of a wxyz quaternion."""
    w, x, y, z = q
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])


def slerp(q0: np.ndarray, q1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1)
    q1 = np.where((dot < 0.0)[..., None], -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    linear = sin_theta < 1e-8
    a = np.empty_like(theta)
    b = np.empty_like(theta)
    np.divide(
        np.sin((1.0 - blend) * theta), sin_theta, out=a, where=~linear
    )
    np.divide(np.sin(blend * theta), sin_theta, out=b, where=~linear)
    a[linear] = (1.0 - blend)[linear]
    b[linear] = blend[linear]
    return normalize_quat(a[..., None] * q0 + b[..., None] * q1)


def gaussian_filter_nearest(values: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    radius = int(4.0 * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    padded = np.pad(values, [(radius, radius)] + [(0, 0)] * (values.ndim - 1), mode="edge")
    result = np.zeros_like(values)
    for offset, weight in enumerate(weights):
        result += weight * padded[offset : offset + len(values)]
    return result


def load_csv(path: Path, width: int) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    values = np.atleast_2d(values)
    if values.shape[1] != width or not np.isfinite(values).all():
        raise ValueError(f"Invalid array in {path}: {values.shape}, expected (*, {width})")
    return values


def load_motion(path: Path) -> dict[str, np.ndarray]:
    joint_pos = load_csv(path / "joint_pos.csv", 29)
    body_pos = load_csv(path / "body_pos.csv", 42).reshape(-1, 14, 3)
    body_quat = load_csv(path / "body_quat.csv", 56).reshape(-1, 14, 4)
    if len(joint_pos) != len(body_pos) or len(joint_pos) != len(body_quat):
        raise ValueError(f"Frame-count mismatch in {path}")
    return {"joint_pos": joint_pos, "body_pos": body_pos, "body_quat": normalize_quat(body_quat)}


def align_second(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> None:
    first_root_pos = first["body_pos"][-1, 0]
    second_root_pos = second["body_pos"][0, 0].copy()
    # Locomotion clips must be aligned on the ground plane. Applying the full
    # pitch/roll difference to a long root trajectory makes it rise or sink.
    rotation = normalize_quat(
        quat_mul(yaw_quat(first["body_quat"][-1, 0]), quat_conjugate(yaw_quat(second["body_quat"][0, 0])))
    )
    relative = second["body_pos"] - second_root_pos
    second["body_pos"] = first_root_pos + quat_rotate(
        np.broadcast_to(rotation, relative.shape[:-1] + (4,)), relative
    )
    second["body_quat"] = normalize_quat(
        quat_mul(np.broadcast_to(rotation, second["body_quat"].shape), second["body_quat"])
    )


def bridge(first: np.ndarray, second: np.ndarray, frames: int, quaternion: bool = False) -> np.ndarray:
    if frames == 0:
        return np.empty((0,) + first.shape, dtype=np.float64)
    t = np.linspace(0.0, 1.0, frames + 2, dtype=np.float64)[1:-1]
    t = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3
    if quaternion:
        return slerp(
            np.broadcast_to(first, (frames,) + first.shape),
            np.broadcast_to(second, (frames,) + second.shape),
            np.broadcast_to(t[:, None], (frames, first.shape[0])),
        )
    return first + t.reshape((frames,) + (1,) * first.ndim) * (second - first)


def calculate_velocities(joint_pos: np.ndarray, body_pos: np.ndarray, body_quat: np.ndarray):
    joint_vel = np.gradient(joint_pos, axis=0) * FPS
    body_lin_vel = gaussian_filter_nearest(np.gradient(body_pos, axis=0) * FPS)
    delta = normalize_quat(quat_mul(body_quat[1:], quat_conjugate(body_quat[:-1])))
    delta = np.where(delta[..., :1] < 0.0, -delta, delta)
    vec_norm = np.linalg.norm(delta[..., 1:], axis=-1)
    angle = 2.0 * np.arctan2(vec_norm, np.clip(delta[..., 0], 0.0, None))
    axis = np.divide(
        delta[..., 1:], vec_norm[..., None], out=np.zeros_like(delta[..., 1:]),
        where=vec_norm[..., None] > 1e-12,
    )
    body_ang_vel = np.zeros_like(body_pos)
    body_ang_vel[:-1] = axis * angle[..., None] * FPS
    body_ang_vel[-1] = body_ang_vel[-2]
    return joint_vel, body_lin_vel, gaussian_filter_nearest(body_ang_vel)


def save_csv(path: Path, values: np.ndarray, header: list[str]) -> None:
    np.savetxt(
        path, values.reshape(len(values), -1), delimiter=",", header=",".join(header),
        comments="", fmt="%.6f",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge-seconds", type=float, default=0.5)
    args = parser.parse_args()

    bridge_frames = round(args.bridge_seconds * FPS)
    if bridge_frames < 0:
        parser.error("--bridge-seconds must be non-negative")

    first = load_motion(args.first)
    second = load_motion(args.second)
    align_second(first, second)

    joint_pos = np.concatenate(
        [first["joint_pos"], bridge(first["joint_pos"][-1], second["joint_pos"][0], bridge_frames), second["joint_pos"]]
    )
    body_pos = np.concatenate(
        [first["body_pos"], bridge(first["body_pos"][-1], second["body_pos"][0], bridge_frames), second["body_pos"]]
    )
    body_quat = normalize_quat(
        np.concatenate(
            [first["body_quat"], bridge(first["body_quat"][-1], second["body_quat"][0], bridge_frames, quaternion=True), second["body_quat"]]
        )
    )
    joint_vel, body_lin_vel, body_ang_vel = calculate_velocities(joint_pos, body_pos, body_quat)

    arrays = [joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel]
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Merged motion contains NaN or infinity")
    quat_error = float(np.max(np.abs(np.linalg.norm(body_quat, axis=-1) - 1.0)))
    if quat_error > 1e-6:
        raise ValueError(f"Quaternion norm error: {quat_error}")

    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "joint_pos.csv", joint_pos, [f"joint_{i}" for i in range(29)])
    save_csv(args.output / "joint_vel.csv", joint_vel, [f"joint_vel_{i}" for i in range(29)])
    save_csv(args.output / "body_pos.csv", body_pos, [f"body_{i}_{axis}" for i in range(14) for axis in "xyz"])
    save_csv(args.output / "body_quat.csv", body_quat, [f"body_{i}_{axis}" for i in range(14) for axis in "wxyz"])
    save_csv(args.output / "body_lin_vel.csv", body_lin_vel, [f"body_{i}_vel_{axis}" for i in range(14) for axis in "xyz"])
    save_csv(args.output / "body_ang_vel.csv", body_ang_vel, [f"body_{i}_angvel_{axis}" for i in range(14) for axis in "xyz"])

    name = args.output.name
    (args.output / "metadata.txt").write_text(
        f"Metadata for: {name}\n{'=' * 30}\n\n"
        "Body part indexes:\n[ 0  4 10 18  5 11 19  9 16 22 28 17 23 29]\n\n"
        f"Total timesteps: {len(joint_pos)}\n",
        encoding="utf-8",
    )
    metadata = {
        "motion_name": name,
        "first_motion": args.first.as_posix(),
        "second_motion": args.second.as_posix(),
        "fps": FPS,
        "bridge_frames": bridge_frames,
        "bridge_seconds": bridge_frames / FPS,
        "root_alignment": "second clip yaw-aligned to first clip; root position joined without rotating the ground plane",
        "frames": len(joint_pos),
        "duration_seconds": (len(joint_pos) - 1) / FPS,
        "max_quaternion_norm_error": quat_error,
        "max_abs_joint_velocity_rad_s": float(np.max(np.abs(joint_vel))),
    }
    (args.output / "merge_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
