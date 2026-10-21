#!/usr/bin/env python3
"""Build two explicit BONES G1 stand/crouch sequences, without touching deployment.

Run from the repository root. Only writes a new, empty output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import convert_bones_to_sonic_reference as conv
from merge_sonic_reference_motions import calculate_velocities, slerp, yaw_quat


FPS = 50
SQUAT = "g1/csv/230509/squat_001__A362.csv"
WALK = "g1/csv/231024/crouch_walk_forward_and_back_turn_360_to_idle_R_002__A495.csv"


def normalize_origin(root, quat):
    """Translate XY and remove initial yaw; preserve recorded floor height."""
    origin = root[0].copy()
    origin[2] = 0
    rotation = conv.quat_conjugate(yaw_quat(quat[0]))
    return conv.quat_rotate(rotation, root - origin), conv.quat_mul(rotation, quat)


def load(path, kin):
    root, quat, joints = conv.read_bones_csv(path, kin)
    root, quat, joints = conv.resample(root, quat, joints, 120, FPS)
    root, quat = normalize_origin(root, quat)
    return {"root": root, "quat": quat, "joints": joints}


def slice_motion(motion, start, stop):
    return {key: value[start:stop].copy() for key, value in motion.items()}


def crossfade(first, second, count):
    """Overlap recorded motion; blend roots/joints and then recompute all FK.

    Alignment is yaw + XY ONLY: aligning absolute root Z would lift/sink feet.
    Quintic weights have zero first and second derivatives at both ends.
    """
    i = len(first["root"]) - count
    rotation = conv.quat_mul(yaw_quat(first["quat"][i]),
                             conv.quat_conjugate(yaw_quat(second["quat"][0])))
    second = {key: value.copy() for key, value in second.items()}
    second["root"] = conv.quat_rotate(rotation, second["root"])
    offset = first["root"][i] - second["root"][0]
    offset[2] = 0
    second["root"] += offset
    second["quat"] = conv.quat_mul(rotation, second["quat"])
    t = np.linspace(0, 1, count)
    weight = 6*t**5 - 15*t**4 + 10*t**3
    result = {}
    for key in first:
        a, b = first[key][i:], second[key][:count]
        middle = slerp(a, b, weight) if key == "quat" else a*(1-weight[:, None])+b*weight[:, None]
        result[key] = np.concatenate([first[key][:i], middle, second[key][count:]])
    return result


def add_holds(motion, frames=50):
    return {key: np.concatenate([np.repeat(value[:1], frames, axis=0), value,
                                 np.repeat(value[-1:], frames, axis=0)])
            for key, value in motion.items()}


def save_and_validate(folder, motion, kin, provenance):
    root, quat, joints = (motion[k] for k in ("root", "quat", "joints"))
    body_pos, body_quat = kin.forward(root, quat, joints)
    joint_pos = np.empty_like(joints)
    joint_pos[:, conv.MJ_TO_IL] = joints
    joint_vel, body_lin_vel, body_ang_vel = calculate_velocities(joint_pos, body_pos, body_quat)
    arrays = dict(joint_pos=joint_pos, joint_vel=joint_vel, body_pos=body_pos,
                  body_quat=body_quat, body_lin_vel=body_lin_vel, body_ang_vel=body_ang_vel)
    count = len(root)
    assert all(len(a) == count and np.isfinite(a).all() for a in arrays.values())
    assert joint_pos.shape == joint_vel.shape == (count, 29)
    assert np.max(abs(np.linalg.norm(body_quat, axis=-1)-1)) < 1e-6
    limits = np.asarray(kin.joint_limits)
    violation = float(np.maximum(np.maximum(limits[:, 0]-joints, joints-limits[:, 1]), 0).max())
    assert violation < np.deg2rad(1), f"Joint limit violation: {violation} rad"
    assert root[0, 2] > .74 and root[-1, 2] > .74, "Must start and finish upright"
    assert root[:, 2].min() < .58, "No actual crouch found"
    assert np.max(np.abs(np.rad2deg(joints[[0,-1]][:, [3,9]]))) < 25, "End knees not straight"
    assert np.max(np.abs(joint_vel)) < 15, "Implausible joint velocity"
    folder.mkdir(parents=True)
    for key, values in arrays.items():
        if key == "joint_pos": header = [f"joint_{i}" for i in range(29)]
        elif key == "joint_vel": header = [f"joint_vel_{i}" for i in range(29)]
        else:
            axes = "wxyz" if key == "body_quat" else "xyz"
            header = [f"body_{i}_{axis}" for i in range(14) for axis in axes]
        conv.save_csv(folder/f"{key}.csv", values, header)
    (folder/"metadata.txt").write_text(
        f"Metadata for: {folder.name}\n==============================\n\n"
        f"Body part indexes:\n{np.array2string(conv.BODY_INDEXES_IL)}\n\n"
        f"Total timesteps: {count}\n", encoding="utf-8")
    rounded = {key: np.loadtxt(folder/f"{key}.csv", delimiter=",", skiprows=1).reshape(value.shape)
               for key, value in arrays.items()}
    assert all(np.isfinite(a).all() for a in rounded.values())
    error = float(abs(np.linalg.norm(rounded["body_quat"], axis=-1)-1).max())
    assert error < 1e-5
    distance = float(np.linalg.norm(root[-1,:2]-root[0,:2]))
    if folder.name.startswith("01"):
        assert distance < .15
    else:
        assert 2.85 < distance < 3.15
    stats = {
        "frames": count, "fps": FPS, "duration_seconds": (count-1)/FPS,
        "root_height_start_min_end_m": [float(root[0,2]),float(root[:,2].min()),float(root[-1,2])],
        "net_displacement_m": distance,
        "xy_path_length_m": float(np.linalg.norm(np.diff(root[:,:2],axis=0),axis=1).sum()),
        "max_joint_velocity_rad_s": float(abs(joint_vel).max()),
        "max_joint_step_rad": float(abs(np.diff(joints,axis=0)).max()),
        "max_joint_limit_violation_rad": violation,
        "quaternion_norm_error_after_csv_rounding": error,
        "knee_start_end_deg": np.rad2deg(joints[[0,-1]][:,[3,9]]).tolist(),
        "ankle_height_range_m": [float(body_pos[:,[3,6],2].min()),float(body_pos[:,[3,6],2].max())],
        "finite": True, "joint_count": 29,
        "physical_policy_tracking_tested": False,
    }
    (folder/"build_metadata.json").write_text(json.dumps({**provenance,"validation":stats},indent=2)+"\n")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=Path("data/bones_seed_full"))
    parser.add_argument("--output",type=Path,default=Path("data/bones_seed_crouch_pair_2026-09-09"))
    args=parser.parse_args()
    if args.output.exists(): parser.error("Output already exists; choose a new directory")
    for source in (SQUAT,WALK):
        if not (args.dataset/source).is_file(): parser.error(f"Missing exact source: {source}")
    kin=conv.G1Kinematics(Path("gear_sonic_deploy/g1/g1_29dof.xml"))
    squat=load(args.dataset/SQUAT,kin)
    walk=load(args.dataset/WALK,kin)
    # First complete native squat only: source time 0..2.30 s (resampled).
    one=add_holds(slice_motion(squat,0,116))
    # Native first descent + outward walk. Overlap starts at 6.68 s.
    # Remove the original return walk and 360-degree turn, then use the
    # original final stand-up from the SAME actor/take (15.86 s onward).
    two=add_holds(crossfade(slice_motion(walk,0,364),slice_motion(walk,793,len(walk["root"])),30))
    common={"source_fps":120,"output_fps":FPS,"start_end_hold_seconds":1,
            "joint_order":"IsaacLab, 29 DOF","quaternion_order":"wxyz",
            "conversion":"cm to m; degrees to radians; extrinsic xyz Euler to quaternion; 120 to 50 Hz",
            "alignment":"initial yaw zero and XY origin zero; original root Z preserved",
            "body_transforms":"recomputed from root + joints through deployment G1 forward kinematics"}
    result={}
    for name,motion,source,extra in [
        ("01_stand_crouch_stand_A362",one,SQUAT,{"actor":"A362","source_ranges_at_50hz_inclusive":[[0,115]],"crossfade_seconds":0}),
        ("02_stand_crouch_walk_3m_stand_A495",two,WALK,{"actor":"A495","source_ranges_at_50hz_inclusive":[[0,363],[793,len(walk["root"])-1]],"crossfade_frames":30,"crossfade_seconds":.6}),
    ]:
        meta={**common,**extra,"source_path":source,"source_sha256":hashlib.sha256((args.dataset/source).read_bytes()).hexdigest()}
        result[name]=save_and_validate(args.output/"ready"/name,motion,kin,meta)
    (args.output/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
