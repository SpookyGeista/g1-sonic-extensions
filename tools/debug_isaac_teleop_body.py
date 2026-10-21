#!/usr/bin/env python3
"""Print raw Isaac Teleop full-body samples without running SONIC/retargeting.

Run from the repository root with ``.venv_teleop`` activated.  The values in
the table are read directly from FullBodyTrackerPico; the final comparison
shows what GR00T's current ``_body_data_to_24x7`` converter produces.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from gear_sonic.utils.teleop.input_readers import _body_data_to_24x7
from gear_sonic.utils.teleop.isaac_teleop_client import IsaacTeleopClient


JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)


def _joint_rows(body_data: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    joints = getattr(body_data, "joints", None)
    getter = getattr(joints, "joints", None)
    if not callable(getter):
        raise TypeError(
            f"Expected FullBodyPosePicoT.joints.joints(index), got {type(body_data).__name__}"
        )

    for index, name in enumerate(JOINT_NAMES):
        joint = getter(index)
        pose = joint.pose
        position = pose.position
        orientation = pose.orientation
        xyz = [float(position.x), float(position.y), float(position.z)]
        xyzw = [
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        ]
        rows.append(
            {
                "index": index,
                "name": name,
                "is_valid": bool(joint.is_valid),
                "position_xyz": xyz,
                "quaternion_xyzw": xyzw,
                "quaternion_norm": math.sqrt(sum(value * value for value in xyzw)),
            }
        )
    return rows


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print(
        "idx  name             valid        x         y         z"
        "        qx        qy        qz        qw    |q|"
    )
    for row in rows:
        x, y, z = row["position_xyz"]
        qx, qy, qz, qw = row["quaternion_xyzw"]
        print(
            f"{row['index']:>3}  {row['name']:<16} {str(row['is_valid']):<5} "
            f"{x:>9.4f} {y:>9.4f} {z:>9.4f} "
            f"{qx:>9.4f} {qy:>9.4f} {qz:>9.4f} {qw:>9.4f} "
            f"{row['quaternion_norm']:>6.3f}"
        )


def _print_checks(rows: list[dict[str, Any]], converted: np.ndarray | None) -> None:
    invalid = [row["index"] for row in rows if not row["is_valid"]]
    print(f"valid joints: {24 - len(invalid)}/24; invalid indices: {invalid or 'none'}")

    bone_lengths = []
    for row, parent in zip(rows, PARENTS):
        if parent < 0 or not row["is_valid"] or not rows[parent]["is_valid"]:
            continue
        child_xyz = np.asarray(row["position_xyz"])
        parent_xyz = np.asarray(rows[parent]["position_xyz"])
        bone_lengths.append((float(np.linalg.norm(child_xyz - parent_xyz)), row["index"], parent))
    if bone_lengths:
        length, child, parent = max(bone_lengths)
        print(
            f"longest valid parent edge: {length:.3f} m "
            f"({parent}:{JOINT_NAMES[parent]} -> {child}:{JOINT_NAMES[child]})"
        )

    if converted is None:
        print("GR00T converter: None")
        return
    zero_rows = np.flatnonzero(np.all(converted == 0.0, axis=1)).tolist()
    zero_quats = np.flatnonzero(np.linalg.norm(converted[:, 3:], axis=1) < 1e-6).tolist()
    print(f"GR00T converter zero-filled rows: {zero_rows or 'none'}")
    print(f"GR00T converter zero-norm quaternions: {zero_quats or 'none'}")


def _print_schema() -> None:
    from isaacteleop.schema import BodyJointPico

    print(f"isaacteleop version: {importlib.metadata.version('isaacteleop')}")
    print("installed BodyJointPico enum:")
    for name, member in BodyJointPico.__members__.items():
        if name != "NUM_JOINTS":
            print(f"  {int(member):>2}  {name.lower()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10, help="number of tables to print")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between tables")
    parser.add_argument("--use-adb", action="store_true")
    parser.add_argument("--jsonl", type=Path, help="optional raw sample output path")
    args = parser.parse_args()

    _print_schema()
    client = IsaacTeleopClient(use_adb=args.use_adb)
    output = args.jsonl.open("a", encoding="utf-8") if args.jsonl else None
    try:
        client.start_streaming()
        received = 0
        print("Waiting for raw FullBodyTrackerPico data (Ctrl+C to stop)...")
        while received < args.samples:
            raw = client._get_tracker_data()  # diagnostic intentionally reads the raw API
            body_data = raw.get("full_body") if raw else None
            if body_data is None:
                time.sleep(0.1)
                continue

            rows = _joint_rows(body_data)
            print(f"\n=== raw sample {received} ===")
            _print_rows(rows)
            _print_checks(rows, _body_data_to_24x7(body_data))
            if output:
                output.write(json.dumps({"sample": received, "joints": rows}) + "\n")
                output.flush()
            received += 1
            time.sleep(max(0.0, args.interval))
    except KeyboardInterrupt:
        pass
    finally:
        if output:
            output.close()
        client.close()


if __name__ == "__main__":
    main()
