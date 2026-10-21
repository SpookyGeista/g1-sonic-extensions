#!/usr/bin/env python3
"""Convert BONES-SEED G1 CSVs to gear_sonic_deploy reference folders.

The conversion follows the repository's deployment contract in
``motion_data_reader.hpp`` and the G1 kinematics in ``fk.cpp``.  BONES input
is 120 Hz, centimeters, extrinsic XYZ Euler degrees, and MuJoCo joint order.
Deployment output is 50 Hz, meters, wxyz quaternions, radians, and IsaacLab
joint order, with 14 selected body transforms and finite velocities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


SOURCE_FPS = 120.0
TARGET_FPS = 50.0
NUM_JOINTS = 29

# For MuJoCo joint index mj, MJ_TO_IL[mj] is its IsaacLab index.
MJ_TO_IL = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
     11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
IL_TO_MJ = np.argsort(MJ_TO_IL)

# Exact 14-body selection stored in the checked-in SONIC reference metadata.
BODY_INDEXES_IL = np.array([0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29])
BODY_INDEXES_MJ = np.array(
    [0 if index == 0 else IL_TO_MJ[index - 1] + 1 for index in BODY_INDEXES_IL]
)

ROOT_COLUMNS = [
    "root_translateX", "root_translateY", "root_translateZ",
    "root_rotateX", "root_rotateY", "root_rotateZ",
]


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


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


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    result = q.copy()
    result[..., 1:] *= -1.0
    return result


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def angle_axis_quat(angle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    half = angle * 0.5
    return np.concatenate([np.cos(half)[..., None], np.sin(half)[..., None] * axis], axis=-1)


def euler_extrinsic_xyz_to_wxyz(euler_rad: np.ndarray) -> np.ndarray:
    """Match scipy Rotation.from_euler('xyz'): extrinsic X, then Y, then Z."""
    shape = euler_rad.shape[:-1]
    basis = np.eye(3)
    qx = angle_axis_quat(euler_rad[..., 0], np.broadcast_to(basis[0], shape + (3,)))
    qy = angle_axis_quat(euler_rad[..., 1], np.broadcast_to(basis[1], shape + (3,)))
    qz = angle_axis_quat(euler_rad[..., 2], np.broadcast_to(basis[2], shape + (3,)))
    return quat_normalize(quat_mul(qz, quat_mul(qy, qx)))


def slerp(q0: np.ndarray, q1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1)
    q1 = np.where((dot < 0.0)[..., None], -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    linear = sin_theta < 1e-7
    a = np.where(linear, 1.0 - blend, np.sin((1.0 - blend) * theta) / sin_theta)
    b = np.where(linear, blend, np.sin(blend * theta) / sin_theta)
    return quat_normalize(a[..., None] * q0 + b[..., None] * q1)


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


class G1Kinematics:
    """Numpy implementation of gear_sonic_deploy's RobotFK."""

    def __init__(self, mjcf_path: Path):
        root = ET.parse(mjcf_path).getroot().find("worldbody/body")
        if root is None:
            raise ValueError(f"No root body in {mjcf_path}")
        self.names: list[str] = []
        self.parents: list[int] = []
        self.offsets: list[np.ndarray] = []
        self.rest_quats: list[np.ndarray] = []
        self.axes: list[np.ndarray] = []
        self.joint_names: list[str] = []
        self.joint_limits: list[np.ndarray] = []
        self._add_body(root, -1)
        if len(self.names) != NUM_JOINTS + 1:
            raise ValueError(f"Expected 30 G1 bodies, found {len(self.names)}")

    def _add_body(self, body: ET.Element, parent: int) -> None:
        index = len(self.names)
        self.names.append(body.attrib["name"])
        self.parents.append(parent)
        self.offsets.append(np.fromstring(body.attrib.get("pos", "0 0 0"), sep=" "))
        self.rest_quats.append(
            quat_normalize(np.fromstring(body.attrib.get("quat", "1 0 0 0"), sep=" "))
        )
        joint = body.find("joint")
        if joint is None:
            raise ValueError(f"Body {body.attrib['name']} has no joint")
        self.axes.append(np.fromstring(joint.attrib.get("axis", "0 0 0"), sep=" "))
        if index > 0:
            self.joint_names.append(joint.attrib["name"])
            self.joint_limits.append(np.fromstring(joint.attrib["range"], sep=" "))
        for child in body.findall("body"):
            self._add_body(child, index)

    def forward(self, root_pos: np.ndarray, root_quat: np.ndarray, dof_mj: np.ndarray):
        frames = len(root_pos)
        positions = np.zeros((frames, len(self.names), 3), dtype=np.float64)
        rotations = np.zeros((frames, len(self.names), 4), dtype=np.float64)
        positions[:, 0] = root_pos
        rotations[:, 0] = root_quat
        offsets = np.asarray(self.offsets)
        rest = np.asarray(self.rest_quats)
        axes = np.asarray(self.axes)
        for child in range(1, len(self.names)):
            parent = self.parents[child]
            positions[:, child] = positions[:, parent] + quat_rotate(
                rotations[:, parent], np.broadcast_to(offsets[child], (frames, 3))
            )
            rotated_axis = quat_rotate(rest[child], axes[child])
            local = quat_mul(
                angle_axis_quat(dof_mj[:, child - 1], np.broadcast_to(rotated_axis, (frames, 3))),
                rest[child],
            )
            rotations[:, child] = quat_normalize(quat_mul(rotations[:, parent], local))
        return positions[:, BODY_INDEXES_MJ], rotations[:, BODY_INDEXES_MJ]


def read_bones_csv(path: Path, kinematics: G1Kinematics):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    joint_columns = [column for column in columns if column.endswith("_dof")]
    expected = kinematics.joint_names
    expected_dof = [f"{name}_dof" for name in expected]
    if columns[:7] != ["Frame", *ROOT_COLUMNS] or joint_columns != expected_dof:
        raise ValueError(f"{path}: schema or MuJoCo joint order does not match deployment MJCF")
    if len(joint_columns) != NUM_JOINTS or not rows:
        raise ValueError(f"{path}: expected non-empty 29-DOF data")
    frame_index = np.array([int(row["Frame"]) for row in rows])
    if not np.array_equal(frame_index, np.arange(len(rows))):
        raise ValueError(f"{path}: Frame must be contiguous and zero-based")
    root_pos_cm = np.array([[float(row[c]) for c in ROOT_COLUMNS[:3]] for row in rows])
    root_euler_deg = np.array([[float(row[c]) for c in ROOT_COLUMNS[3:]] for row in rows])
    dof_deg = np.array([[float(row[c]) for c in joint_columns] for row in rows])
    return root_pos_cm / 100.0, euler_extrinsic_xyz_to_wxyz(np.deg2rad(root_euler_deg)), np.deg2rad(dof_deg)


def resample(root_pos, root_quat, dof_mj, source_fps, target_fps):
    if source_fps == target_fps:
        return root_pos, root_quat, dof_mj
    duration = (len(root_pos) - 1) / source_fps
    target_times = np.arange(0.0, duration, 1.0 / target_fps)
    source_position = target_times * source_fps
    index0 = np.floor(source_position).astype(int)
    index1 = np.minimum(index0 + 1, len(root_pos) - 1)
    blend = source_position - index0
    root_pos_out = root_pos[index0] * (1.0 - blend[:, None]) + root_pos[index1] * blend[:, None]
    dof_out = dof_mj[index0] * (1.0 - blend[:, None]) + dof_mj[index1] * blend[:, None]
    root_quat_out = slerp(root_quat[index0], root_quat[index1], blend)
    return root_pos_out, root_quat_out, dof_out


def calculate_velocities(joint_pos, body_pos, body_quat, fps):
    joint_vel = np.diff(joint_pos, axis=0) * fps
    joint_vel = np.concatenate([joint_vel, joint_vel[-2:-1]], axis=0)
    body_lin_vel = gaussian_filter_nearest(np.gradient(body_pos, axis=0) * fps)
    delta = quat_normalize(quat_mul(body_quat[1:], quat_conjugate(body_quat[:-1])))
    # q and -q describe the same rotation; select the shortest increment.
    delta = np.where((delta[..., :1] < 0.0), -delta, delta)
    vec_norm = np.linalg.norm(delta[..., 1:], axis=-1)
    angle = 2.0 * np.arctan2(vec_norm, np.clip(delta[..., 0], 0.0, None))
    axis = np.divide(
        delta[..., 1:], vec_norm[..., None], out=np.zeros_like(delta[..., 1:]),
        where=vec_norm[..., None] > 1e-12,
    )
    body_ang_vel = np.zeros_like(body_pos)
    body_ang_vel[:-1] = axis * angle[..., None] * fps
    body_ang_vel = gaussian_filter_nearest(body_ang_vel)
    return joint_vel, body_lin_vel, body_ang_vel


def save_csv(path: Path, values: np.ndarray, header: list[str]) -> None:
    np.savetxt(path, values.reshape(len(values), -1), delimiter=",", header=",".join(header), comments="", fmt="%.6f")


def validate_motion(data: dict, limits_il: np.ndarray, source_frames: int) -> dict:
    arrays = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("NaN or infinity in converted motion")
    frames = len(data["joint_pos"])
    if data["joint_pos"].shape != (frames, NUM_JOINTS) or data["joint_vel"].shape != (frames, NUM_JOINTS):
        raise ValueError("Converted motion does not contain 29 joints per frame")
    quat_error = float(np.max(np.abs(np.linalg.norm(data["body_quat"], axis=-1) - 1.0)))
    if quat_error > 1e-5:
        raise ValueError(f"Quaternion norm error {quat_error}")
    duration = (frames - 1) / TARGET_FPS
    source_duration = (source_frames - 1) / SOURCE_FPS
    if frames < 2 or abs(duration - source_duration) > 1.0 / TARGET_FPS + 1e-9:
        raise ValueError("Implausible duration or inconsistent FPS")
    root_height = data["body_pos"][:, 0, 2]
    if root_height.min() < 0.15 or root_height.max() > 1.5:
        raise ValueError(f"Implausible G1 root height [{root_height.min()}, {root_height.max()}]")
    below = np.maximum(limits_il[:, 0] - data["joint_pos"], 0.0)
    above = np.maximum(data["joint_pos"] - limits_il[:, 1], 0.0)
    max_limit_violation = float(np.maximum(below, above).max())
    return {
        "source_frames": source_frames,
        "output_frames": frames,
        "source_duration_s": source_duration,
        "output_duration_s": duration,
        "root_height_min_m": float(root_height.min()),
        "root_height_max_m": float(root_height.max()),
        "max_quaternion_norm_error": quat_error,
        "max_abs_joint_velocity_rad_s": float(np.abs(data["joint_vel"]).max()),
        "max_joint_limit_violation_rad": max_limit_violation,
        "joint_limits_ok": max_limit_violation <= math.radians(1.0),
    }


def convert_one(input_path: Path, output_dir: Path, kinematics: G1Kinematics) -> dict:
    root_pos, root_quat, dof_mj = read_bones_csv(input_path, kinematics)
    source_frames = len(root_pos)
    root_pos, root_quat, dof_mj = resample(root_pos, root_quat, dof_mj, SOURCE_FPS, TARGET_FPS)
    dof_il = np.empty_like(dof_mj)
    dof_il[:, MJ_TO_IL] = dof_mj
    body_pos, body_quat = kinematics.forward(root_pos, root_quat, dof_mj)
    joint_vel, body_lin_vel, body_ang_vel = calculate_velocities(dof_il, body_pos, body_quat, TARGET_FPS)
    data = {
        "joint_pos": dof_il,
        "joint_vel": joint_vel,
        "body_pos": body_pos,
        "body_quat": body_quat,
        "body_lin_vel": body_lin_vel,
        "body_ang_vel": body_ang_vel,
    }
    limits_mj = np.asarray(kinematics.joint_limits)
    limits_il = np.empty_like(limits_mj)
    limits_il[MJ_TO_IL] = limits_mj
    validation = validate_motion(data, limits_il, source_frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "joint_pos.csv", dof_il, [f"joint_{i}" for i in range(NUM_JOINTS)])
    save_csv(output_dir / "joint_vel.csv", joint_vel, [f"joint_vel_{i}" for i in range(NUM_JOINTS)])
    save_csv(output_dir / "body_pos.csv", body_pos, [f"body_{i}_{axis}" for i in range(14) for axis in "xyz"])
    save_csv(output_dir / "body_quat.csv", body_quat, [f"body_{i}_{axis}" for i in range(14) for axis in "wxyz"])
    save_csv(output_dir / "body_lin_vel.csv", body_lin_vel, [f"body_{i}_vel_{axis}" for i in range(14) for axis in "xyz"])
    save_csv(output_dir / "body_ang_vel.csv", body_ang_vel, [f"body_{i}_angvel_{axis}" for i in range(14) for axis in "xyz"])

    name = input_path.stem
    metadata = (
        f"Metadata for: {name}\n{'=' * 30}\n\n"
        f"Body part indexes:\n{np.array2string(BODY_INDEXES_IL, separator=' ')}\n\n"
        f"Total timesteps: {len(dof_il)}\n"
    )
    (output_dir / "metadata.txt").write_text(metadata, encoding="utf-8")
    conversion_metadata = {
        "motion_name": name,
        "source_file": input_path.as_posix(),
        "source_fps": SOURCE_FPS,
        "target_fps": TARGET_FPS,
        "source_translation_units": "centimeters",
        "output_translation_units": "meters",
        "source_angle_units": "degrees",
        "output_angle_units": "radians",
        "source_root_rotation": "extrinsic XYZ Euler",
        "output_quaternion_order": "wxyz",
        "source_joint_order": "MuJoCo/MJCF",
        "output_joint_order": "IsaacLab",
        "source_joint_names": kinematics.joint_names,
        "output_joint_names": [
            kinematics.joint_names[IL_TO_MJ[index]] for index in range(NUM_JOINTS)
        ],
        "body_part_indexes_isaaclab": BODY_INDEXES_IL.tolist(),
        "validation": validation,
    }
    (output_dir / "conversion_metadata.json").write_text(
        json.dumps(conversion_metadata, indent=2) + "\n", encoding="utf-8"
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path, help="One BONES G1 CSV")
    mode.add_argument("--manifest", type=Path, help="Selection manifest for batch conversion")
    parser.add_argument("--output", type=Path, help="Output folder for --input")
    parser.add_argument("--output-root", type=Path, default=Path("data/bones_seed_selected/sonic_reference"))
    parser.add_argument("--mjcf", type=Path, default=Path("gear_sonic_deploy/g1/g1_29dof.xml"))
    args = parser.parse_args()
    kinematics = G1Kinematics(args.mjcf)

    if args.input:
        if args.output is None:
            parser.error("--input requires --output")
        jobs = [(args.input, args.output)]
    else:
        with args.manifest.open(encoding="utf-8", newline="") as handle:
            jobs = [
                (Path(row["local_path"]), args.output_root / row["motion_name"])
                for row in csv.DictReader(handle)
            ]

    failures = []
    results = []
    for input_path, output_dir in jobs:
        try:
            result = convert_one(input_path, output_dir, kinematics)
            results.append({"motion_name": input_path.stem, **result})
            print(f"OK {input_path.stem}: {result}")
        except Exception as error:  # noqa: BLE001
            failures.append((input_path, error))
            print(f"FAILED {input_path}: {error}")
    if failures:
        raise SystemExit(f"{len(failures)} conversion(s) failed")
    if args.manifest:
        args.output_root.mkdir(parents=True, exist_ok=True)
        with (args.output_root / "validation.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
