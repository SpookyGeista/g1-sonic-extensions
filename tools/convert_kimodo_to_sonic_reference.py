#!/usr/bin/env python3
"""Convert native Kimodo G1 qpos CSV/NPZ into 50 Hz SONIC references.

This is NOT the BONES CSV parser: Kimodo qpos already uses meters, radians,
Z-up/+X-forward and wxyz quaternions. No unit, root-axis, or joint-rest-pose
conversion is performed. FPS must come from a sidecar or --source-fps.

Example:
  .venv_sim/bin/python tools/convert_kimodo_to_sonic_reference.py \
    --input generated/walk.csv --output generated/sonic/walk

Batch (reads each <stem>.meta.json, which must declare fps):
  .venv_sim/bin/python tools/convert_kimodo_to_sonic_reference.py \
    --input-dir generated/native --glob '**/*.csv' --output-root generated/sonic

NPZ requires the official Kimodo package; its exporter runs on CPU. CSV only
requires numpy and the Kimodo source XML for independent schema verification.
Generated references are kinematic data, NOT a dynamic simulation approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

from convert_bones_to_sonic_reference import (
    BODY_INDEXES_IL, G1Kinematics, IL_TO_MJ, NUM_JOINTS,
    calculate_velocities, quat_normalize, save_csv,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_KIMODO = REPO / "data/kimodo_toolchain_2026-09-16/kimodo"
DEFAULT_MJCF = REPO / "gear_sonic_deploy/g1/g1_29dof.xml"
TARGET_FPS = 50.0  # motion_data_reader.hpp::ComputeGlobalVelocities; deploy control rate
EXPECTED_CONVENTIONS = {
    "root_quaternion_order": "wxyz",
    "translation_units": "meters",
    "joint_angle_units": "radians",
    "coordinate_system": "z_up_x_forward",
    "mujoco_rest_zero": False,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_skeletons(kimodo_root: Path, mjcf_path: Path, fk: G1Kinematics) -> dict:
    """Check the actual exporter XML, including angular conventions, not just names."""
    source_xml = kimodo_root / "kimodo/assets/skeletons/g1skel34/xml/g1.xml"
    source = ET.parse(source_xml).getroot()
    target = ET.parse(mjcf_path).getroot()
    if source.find("compiler").get("angle") != "radian":
        raise ValueError("Kimodo G1 XML must use radians")
    source_joints = source.find("worldbody").findall(".//joint")
    names = [j.get("name") for j in source_joints]
    if names != fk.joint_names or len(names) != NUM_JOINTS:
        raise ValueError("Kimodo XML joint names/order do not match deployment G1")
    source_parent = {child: parent for parent in source.iter() for child in parent}
    target_parent = {child: parent for parent in target.iter() for child in parent}
    target_joints = {j.get("name"): j for j in target.find("worldbody").findall(".//joint")}
    defaults = {
        e.get("class"): e.find("joint").attrib
        for e in source.findall(".//default")
        if e.get("class") and e.find("joint") is not None
    }
    offset_differences = []
    for joint in source_joints:
        name = joint.get("name")
        other = target_joints[name]
        source_axis = np.fromstring(joint.get("axis", defaults.get(joint.get("class"), {}).get("axis", "0 0 1")), sep=" ")
        target_axis = np.fromstring(other.get("axis", "0 0 1"), sep=" ")
        if not np.allclose(source_axis, target_axis, atol=1e-8, rtol=0):
            raise ValueError(f"Hinge-axis mismatch for {name}; explicit retargeting required")
        source_body, target_body = source_parent[joint], target_parent[other]
        sq = quat_normalize(np.fromstring(source_body.get("quat", "1 0 0 0"), sep=" "))
        tq = quat_normalize(np.fromstring(target_body.get("quat", "1 0 0 0"), sep=" "))
        if 1.0 - abs(float(np.dot(sq, tq))) > 1e-10:
            raise ValueError(f"Rest-rotation mismatch for {name}; explicit retargeting required")
        sp = np.fromstring(source_body.get("pos", "0 0 0"), sep=" ")
        tp = np.fromstring(target_body.get("pos", "0 0 0"), sep=" ")
        if not np.allclose(sp, tp, atol=1e-8, rtol=0):
            offset_differences.append({"joint": name, "kimodo_m": sp.tolist(), "deployment_m": tp.tolist()})
    return {
        "kimodo_xml": str(source_xml.resolve()), "kimodo_xml_sha256": sha256(source_xml),
        "deployment_xml": str(mjcf_path.resolve()), "deployment_xml_sha256": sha256(mjcf_path),
        "joint_names_order_axes_and_rest_rotations_match": True,
        "morphology_offset_differences": offset_differences,
        "fk_morphology": "deployment XML, not Kimodo XML",
    }


def read_metadata(path: Path, metadata_path: Path | None = None) -> tuple[dict, Path | None]:
    selected = metadata_path or path.with_suffix(".meta.json")
    if not selected.exists():
        if metadata_path is not None:
            raise FileNotFoundError(selected)
        return {}, None
    metadata = json.loads(selected.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Source metadata must be a JSON object")
    for key, expected in EXPECTED_CONVENTIONS.items():
        if key in metadata and metadata[key] != expected:
            raise ValueError(f"Unsupported {key}={metadata[key]!r}; expected {expected!r}")
    return metadata, selected


def resolve_fps(metadata: dict, explicit_fps: float | None, embedded_fps: float | None = None) -> tuple[float, list[str]]:
    candidates = []
    if explicit_fps is not None:
        candidates.append(("--source-fps", explicit_fps))
    for key in ("fps", "source_fps", "mocap_frame_rate"):
        if key in metadata:
            candidates.append((f"sidecar:{key}", metadata[key]))
    if embedded_fps is not None:
        candidates.append(("NPZ:fps", embedded_fps))
    if not candidates:
        raise ValueError("Unknown source FPS: provide <stem>.meta.json with fps or --source-fps; no 30/120 Hz guess is allowed")
    values = [float(value) for _, value in candidates]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Source FPS must be finite and positive")
    if any(not math.isclose(value, values[0], rel_tol=1e-8, abs_tol=1e-8) for value in values):
        raise ValueError(f"Conflicting source FPS evidence: {candidates}")
    return values[0], [name for name, _ in candidates]


def load_qpos(path: Path, kimodo_root: Path) -> tuple[np.ndarray, float | None, str]:
    if path.suffix.lower() == ".csv":
        return np.loadtxt(path, delimiter=",", ndmin=2), None, "native headerless MuJoCo qpos CSV"
    if path.suffix.lower() != ".npz":
        raise ValueError("Input must be a native Kimodo G1 .csv or .npz")
    with np.load(path, allow_pickle=False) as archive:
        raw = {key: np.asarray(archive[key]) for key in archive.files}
        fps_values = [float(np.asarray(archive[key]).item()) for key in ("fps", "mocap_frame_rate") if key in archive]
    if fps_values and any(not math.isclose(x, fps_values[0]) for x in fps_values):
        raise ValueError("Conflicting embedded NPZ FPS values")
    # Official exporter handles Kimodo's Y-up/+Z-forward skeleton projection.
    # Do not reimplement Euler projection or import the BONES root conversion.
    sys.path.insert(0, str(kimodo_root.resolve()))
    import torch
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.skeleton import global_rots_to_local_rots
    from kimodo.skeleton.registry import build_skeleton
    skeleton = build_skeleton(34, kimodo_root / "kimodo/assets/skeletons")
    if "local_rot_mats" in raw:
        local = raw["local_rot_mats"]
    elif "global_rot_mats" in raw:
        # Some official bundled examples only contain global transforms.
        global_rots = raw["global_rot_mats"]
        if global_rots.shape[-3:] != (34, 3, 3):
            raise ValueError("Expected G1 global rotations with 34 joints")
        local = global_rots_to_local_rots(torch.as_tensor(global_rots, device="cpu"), skeleton).numpy()
    else:
        raise ValueError("NPZ needs local_rot_mats or global_rot_mats")
    if "root_positions" in raw:
        root = raw["root_positions"]
    elif "posed_joints" in raw and raw["posed_joints"].shape[-2:] == (34, 3):
        root = raw["posed_joints"][..., skeleton.root_idx, :]
    else:
        raise ValueError("NPZ needs root_positions or posed_joints for 34-joint G1")
    if local.ndim == 4:
        local = local[None]
    if root.ndim == 2:
        root = root[None]
    if local.ndim != 5 or local.shape[0] != 1 or local.shape[2:] != (34, 3, 3):
        raise ValueError(f"Expected one G1 sample with 34 local rotation matrices, found {local.shape}")
    if root.shape != (1, local.shape[1], 3) or not np.isfinite(local).all() or not np.isfinite(root).all():
        raise ValueError("Invalid root positions or non-finite NPZ data")
    ortho = np.matmul(np.swapaxes(local, -1, -2), local)
    if np.max(np.abs(ortho - np.eye(3))) > 1e-3 or np.max(np.abs(np.linalg.det(local) - 1)) > 1e-3:
        raise ValueError("NPZ local rotations are not proper rotation matrices")
    converter = MujocoQposConverter(skeleton, str(kimodo_root / "kimodo/assets/skeletons/g1skel34/xml/g1.xml"))
    qpos = converter.dict_to_qpos({"local_rot_mats": local.astype(np.float32), "root_positions": root.astype(np.float32)}, device="cpu", mujoco_rest_zero=False)
    return np.asarray(qpos[0]), fps_values[0] if fps_values else None, "native G1 NPZ through official float32 CPU MujocoQposConverter (no clipping)"


def resample_qpos(qpos: np.ndarray, source_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Uniform 50 Hz sampling; retain last exact-grid endpoint, never extrapolate."""
    duration = (len(qpos) - 1) / source_fps
    times = np.arange(int(math.floor(duration * TARGET_FPS + 1e-8)) + 1) / TARGET_FPS
    if len(times) < 3:
        raise ValueError("Motion is too short: require at least 3 output frames")
    source_positions = np.minimum(times * source_fps, len(qpos) - 1)
    i0 = np.floor(source_positions).astype(int)
    i1 = np.minimum(i0 + 1, len(qpos) - 1)
    blend = source_positions - i0
    output = qpos[i0] * (1 - blend[:, None]) + qpos[i1] * blend[:, None]
    q0, q1 = qpos[i0, 3:7], qpos[i1, 3:7].copy()
    dot = np.sum(q0 * q1, axis=-1)
    q1[dot < 0] *= -1
    dot = np.clip(abs(dot), 0, 1)
    theta = np.arccos(dot)
    curved = dot < 0.999999
    a, b = 1 - blend, blend.copy()
    a[curved] = np.sin((1 - blend[curved]) * theta[curved]) / np.sin(theta[curved])
    b[curved] = np.sin(blend[curved] * theta[curved]) / np.sin(theta[curved])
    output[:, 3:7] = quat_normalize(a[:, None] * q0 + b[:, None] * q1)
    return output, times


def limit_report(joints_mj: np.ndarray, fk: G1Kinematics) -> dict:
    limits = np.asarray(fk.joint_limits)
    excess = np.maximum(np.maximum(limits[:, 0] - joints_mj, joints_mj - limits[:, 1]), 0)
    worst_frame, worst_joint = np.unravel_index(np.argmax(excess), excess.shape)
    return {"max_excess_rad": float(excess.max()), "worst_frame": int(worst_frame),
            "worst_joint": fk.joint_names[worst_joint], "frames_with_any_excess": int(np.count_nonzero(np.any(excess > 1e-6, axis=1)))}


def trajectory_summary(qpos: np.ndarray) -> dict:
    root = qpos[:, :3]
    w, x, y, z = qpos[:, 3:7].T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return {
        "start_root_xyz_m": root[0].tolist(), "end_root_xyz_m": root[-1].tolist(),
        "root_displacement_xyz_m": (root[-1] - root[0]).tolist(),
        "root_path_xy_m": float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1).sum()),
        "start_yaw_deg": float(np.rad2deg(yaw[0])), "end_yaw_deg": float(np.rad2deg(yaw[-1])),
        "yaw_change_deg": float(np.rad2deg(yaw[-1] - yaw[0])),
        "start_joint_pos_mujoco_rad": qpos[0, 7:].tolist(),
        "end_joint_pos_mujoco_rad": qpos[-1, 7:].tolist(),
    }


def convert_one(input_path: Path, output_dir: Path, *, source_fps: float | None = None,
                metadata_path: Path | None = None, kimodo_root: Path = DEFAULT_KIMODO,
                mjcf_path: Path = DEFAULT_MJCF, joint_limit_policy: str = "reject",
                joint_limit_tolerance_rad: float = 1e-5) -> dict:
    if joint_limit_policy not in ("reject", "report"):
        raise ValueError("joint_limit_policy must be reject or report")
    if not math.isfinite(joint_limit_tolerance_rad) or joint_limit_tolerance_rad < 0:
        raise ValueError("Joint-limit tolerance must be finite and nonnegative")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output {output_dir}")
    if output_dir.resolve() == input_path.resolve() or output_dir.resolve() in input_path.resolve().parents:
        raise ValueError("Output folder must not contain the original input")
    fk = G1Kinematics(mjcf_path)
    schema = audit_skeletons(kimodo_root, mjcf_path, fk)
    metadata, sidecar = read_metadata(input_path, metadata_path)
    input_hash_key = {".csv": "native_csv_sha256", ".npz": "native_npz_sha256"}.get(input_path.suffix.lower())
    if input_hash_key in metadata and metadata[input_hash_key] != sha256(input_path):
        raise ValueError(f"Source hash mismatch for {input_hash_key}: stale sidecar or modified input")
    if "native_model_xml_sha256" in metadata and metadata["native_model_xml_sha256"] != schema["kimodo_xml_sha256"]:
        raise ValueError("Native Kimodo model XML hash mismatch: exporter skeleton differs from current audit")
    if "joint_names" in metadata and metadata["joint_names"] != fk.joint_names:
        raise ValueError("Sidecar joint_names do not match verified native MuJoCo order")
    qpos, embedded_fps, parse_method = load_qpos(input_path, kimodo_root)
    fps, fps_evidence = resolve_fps(metadata, source_fps, embedded_fps)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2 or not np.isfinite(qpos).all():
        raise ValueError(f"Expected finite [T,36] native qpos, found {qpos.shape}")
    qpos = qpos.astype(np.float64, copy=True)
    quat_error = float(np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=-1) - 1)))
    if quat_error > 1e-3:
        raise ValueError(f"Source quaternion norm error {quat_error}; wrong convention or corrupt data")
    qpos[:, 3:7] = quat_normalize(qpos[:, 3:7])
    source_limits = limit_report(qpos[:, 7:], fk)
    if source_limits["max_excess_rad"] > joint_limit_tolerance_rad and joint_limit_policy == "reject":
        raise ValueError(f"Source joint limit violation: {source_limits}; use --joint-limit-policy report only for explicit simulation review. No joints were clipped.")
    out, times = resample_qpos(qpos, fps)
    dof_il = out[:, 7:][:, IL_TO_MJ]
    body_pos, body_quat = fk.forward(out[:, :3], out[:, 3:7], out[:, 7:])
    velocities = calculate_velocities(dof_il, body_pos, body_quat, TARGET_FPS)
    data = dict(zip(("joint_pos", "body_pos", "body_quat", "joint_vel", "body_lin_vel", "body_ang_vel"),
                    (dof_il, body_pos, body_quat, *velocities)))
    if any(not np.isfinite(v).all() for v in data.values()):
        raise ValueError("Converted references contain NaN/Inf")
    height = out[:, 2]
    if height.min() < 0.15 or height.max() > 1.5:
        raise ValueError(f"Implausible locomotion root height [{height.min()}, {height.max()}] meters")
    output_limits = limit_report(out[:, 7:], fk)
    validation = {
        "source_frames": len(qpos), "output_frames": len(out), "source_fps": fps, "target_fps": TARGET_FPS,
        "source_duration_s": (len(qpos) - 1) / fps, "output_duration_s": float(times[-1]),
        "timestamp_step_s": 1 / TARGET_FPS, "source_quaternion_norm_error": quat_error,
        "output_quaternion_norm_error": float(np.max(abs(np.linalg.norm(body_quat, axis=-1) - 1))),
        "root_height_min_m": float(height.min()), "root_height_max_m": float(height.max()),
        "max_abs_joint_velocity_rad_s": float(abs(velocities[0]).max()),
        "source_joint_limits": source_limits, "output_joint_limits": output_limits,
        "joint_limit_policy": joint_limit_policy, "joint_limit_tolerance_rad": joint_limit_tolerance_rad,
        "joint_limits_ok": max(source_limits["max_excess_rad"], output_limits["max_excess_rad"]) <= joint_limit_tolerance_rad,
        "finite": True, "dynamic_simulation_tested": False,
        "trajectory": trajectory_summary(out),
    }
    headers = {
        "joint_pos": [f"joint_{i}" for i in range(29)], "joint_vel": [f"joint_vel_{i}" for i in range(29)],
        "body_pos": [f"body_{i}_{a}" for i in range(14) for a in "xyz"],
        "body_quat": [f"body_{i}_{a}" for i in range(14) for a in "wxyz"],
        "body_lin_vel": [f"body_{i}_vel_{a}" for i in range(14) for a in "xyz"],
        "body_ang_vel": [f"body_{i}_angvel_{a}" for i in range(14) for a in "xyz"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, values in data.items():
        save_csv(output_dir / f"{key}.csv", values, headers[key])
    (output_dir / "metadata.txt").write_text(
        f"Metadata for: {input_path.stem}\n{'=' * 30}\n\nBody part indexes:\n"
        f"{np.array2string(BODY_INDEXES_IL, separator=' ')}\n\nTotal timesteps: {len(out)}\n", encoding="utf-8")
    provenance = {
        "motion_name": input_path.stem, "source_file": str(input_path.resolve()), "source_sha256": sha256(input_path),
        "source_metadata_file": str(sidecar.resolve()) if sidecar else None,
        "source_metadata_sha256": sha256(sidecar) if sidecar else None, "source_metadata": metadata,
        "source_fps": fps, "source_fps_evidence": fps_evidence, "target_fps": TARGET_FPS,
        "parser": parse_method, "source_conventions": EXPECTED_CONVENTIONS,
        "source_joint_names": fk.joint_names, "output_joint_names": [fk.joint_names[i] for i in IL_TO_MJ],
        "output_joint_order": "IsaacLab", "body_part_indexes_isaaclab": BODY_INDEXES_IL.tolist(),
        "schema_audit": schema,
        "conversion": ["normalize root quaternion roundoff", "linear root/joint interpolation and shortest-arc root SLERP on 50 Hz timestamps (no extrapolation)",
                       "MuJoCo to IsaacLab joint permutation", "deployment G1 FK for 14 body poses",
                       "existing BONES/SONIC exporter velocity calculation: forward joint difference, Gaussian(sigma=2) body velocities"],
        "pose_clipping_or_smoothing": "none", "root_recenter_or_heading_reset": "none",
        "converter_sha256": sha256(Path(__file__)),
        "shared_fk_exporter_sha256": sha256(Path(__file__).with_name("convert_bones_to_sonic_reference.py")),
        "output_csv_sha256": {f"{k}.csv": sha256(output_dir / f"{k}.csv") for k in data},
        "validation": validation,
    }
    (output_dir / "conversion_metadata.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path)
    mode.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--glob", default="*.csv", help="Batch input filename pattern; default *.csv (not recursive)")
    parser.add_argument("--source-fps", type=float, help="Explicit native generation FPS, never guessed")
    parser.add_argument("--source-metadata", type=Path, help="Single-input sidecar; default <stem>.meta.json")
    parser.add_argument("--kimodo-root", type=Path, default=DEFAULT_KIMODO)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--joint-limit-policy", choices=("reject", "report"), default="reject")
    parser.add_argument("--joint-limit-tolerance-rad", type=float, default=1e-5)
    args = parser.parse_args()
    if args.input:
        if args.output is None:
            parser.error("--input requires --output")
        jobs = [(args.input, args.output)]
    else:
        if args.output_root is None or args.source_metadata is not None:
            parser.error("--input-dir requires --output-root and per-input sidecars, not --source-metadata")
        paths = sorted(p for p in args.input_dir.glob(args.glob) if p.is_file())
        if not paths:
            parser.error("No matching inputs; nothing converted")
        jobs = [(p, args.output_root / p.stem) for p in paths]
        if len({out for _, out in jobs}) != len(jobs):
            parser.error("Multiple inputs map to the same output folder")
    failures = 0
    for input_path, output_dir in jobs:
        try:
            result = convert_one(input_path, output_dir, source_fps=args.source_fps, metadata_path=args.source_metadata,
                                 kimodo_root=args.kimodo_root, mjcf_path=args.mjcf, joint_limit_policy=args.joint_limit_policy,
                                 joint_limit_tolerance_rad=args.joint_limit_tolerance_rad)
            print(json.dumps({"input": str(input_path), "output": str(output_dir), "validation": result}))
        except (ValueError, OSError, KeyError, ImportError) as error:
            failures += 1
            print(f"FAILED {input_path}: {error}", file=sys.stderr)
    if failures:
        raise SystemExit(f"{failures}/{len(jobs)} conversion(s) failed")


if __name__ == "__main__":
    main()
