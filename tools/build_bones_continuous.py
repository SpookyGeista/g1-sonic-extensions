#!/usr/bin/env python3
"""Separate BONES reference chain: trim declared idle tails, overlap moving clips.

Never overwrites sources or inserts standing bridges. Native active samples stay
at 50 Hz. Only XY/yaw placement and overlapping reference poses are changed;
this is not contact-constrained optimization and requires physical simulation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
import convert_bones_to_sonic_reference as conv
from build_bones_crouch_pair import crossfade
from build_bones_walk_video_pack import describe, digest, load_reference, save

REPO = Path(__file__).resolve().parents[1]
FPS = 50
CHANNELS = ('joint_pos', 'joint_vel', 'body_pos', 'body_quat', 'body_lin_vel', 'body_ang_vel')


def identical_edges(motion):
    root, quat, joints = (motion[key] for key in ('root', 'quat', 'joints'))
    qdelta = np.minimum(np.linalg.norm(np.diff(quat, axis=0), axis=1),
                        np.linalg.norm(quat[1:] + quat[:-1], axis=1))
    return ((np.max(abs(np.diff(root, axis=0)), axis=1) <= 1e-10)
            & (np.max(abs(np.diff(joints, axis=0)), axis=1) <= 1e-10)
            & (qdelta <= 1e-10))


def collapse_holds(motion, minimum_frames=3):
    """Retain one pose per exact repeated run; do not remove low-speed motion."""
    if minimum_frames < 2:
        raise ValueError('Repeated-pose threshold must be at least two frames')
    keep = np.ones(len(motion['root']), dtype=bool)
    edges = identical_edges(motion)
    runs = []
    index = 0
    while index < len(edges):
        if not edges[index]:
            index += 1
            continue
        stop = index
        while stop < len(edges) and edges[stop]:
            stop += 1
        if stop - index + 1 >= minimum_frames:
            keep[index + 1:stop + 1] = False
            runs.append({'start_frame': index, 'end_frame_exclusive': stop + 1,
                         'removed_frames': stop - index})
        index = stop
    indices = np.flatnonzero(keep)
    return {key: value[keep].copy() for key, value in motion.items()}, indices, runs


def longest_true_run(mask):
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def activity(motion):
    joint_speed = np.diff(motion['joints'], axis=0) * FPS
    root_speed = np.linalg.norm(np.diff(motion['root'], axis=0), axis=1) * FPS
    rms = np.sqrt(np.mean(joint_speed ** 2, axis=1))
    # Diagnostic thresholds, not a physical or perceptual guarantee.
    quiet = (root_speed < .025) & (rms < .08)
    return {'longest_exact_duplicate_interval_s': longest_true_run(identical_edges(motion))/FPS,
            'longest_low_activity_interval_s': longest_true_run(quiet)/FPS,
            'low_activity_definition': 'Root 3D speed <0.025 m/s AND RMS joint speed <0.08 rad/s',
            'low_activity_total_s': float(np.count_nonzero(quiet)/FPS)}


def independent_fk(folder, kin):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(REPO/'gear_sonic_deploy/g1/g1_29dof.xml'))
    data = mujoco.MjData(model)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, kin.names[index])
           for index in conv.BODY_INDEXES_MJ]
    pos = np.loadtxt(folder/'body_pos.csv', delimiter=',', skiprows=1).reshape(-1,14,3)
    quat = np.loadtxt(folder/'body_quat.csv', delimiter=',', skiprows=1).reshape(-1,14,4)
    joints = np.loadtxt(folder/'joint_pos.csv', delimiter=',', skiprows=1)
    max_pos = max_angle = 0.0
    for frame in range(len(pos)):
        data.qpos[:3] = pos[frame,0]
        data.qpos[3:7] = conv.quat_normalize(quat[frame,0])
        data.qpos[7:] = joints[frame, conv.MJ_TO_IL]
        mujoco.mj_forward(model,data)
        max_pos = max(max_pos, float(abs(data.xpos[ids]-pos[frame]).max()))
        reference = conv.quat_normalize(quat[frame])
        dot = abs(np.sum(data.xquat[ids]*reference,axis=-1))
        max_angle = max(max_angle,float(np.rad2deg(2*np.arccos(np.clip(dot,0,1))).max()))
    if max_pos > 5e-6 or max_angle > .001:
        raise ValueError(f'Independent FK mismatch: {max_pos} m / {max_angle} deg')
    return {'passed':True,'frames':len(pos),'max_position_error_m':max_pos,
            'max_orientation_error_deg':max_angle}


def build(spec_path, output, overlap_seconds=.4):
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(REPO):
        raise ValueError('Choose a NEW output directory inside the repository')
    count = round(overlap_seconds*FPS)
    if not np.isfinite(overlap_seconds) or count < 3 or count > 50:
        raise ValueError('Use 3–50 overlap frames at 50 Hz')
    spec = json.loads(spec_path.read_text())
    if spec.get('fps') != FPS or not spec.get('stages'):
        raise ValueError('Expected nonempty 50 Hz sequence')
    kin = conv.G1Kinematics(REPO/'gear_sonic_deploy/g1/g1_29dof.xml')
    hashes, prepared, sources = {}, [], []
    for stage in spec['stages']:
        folder = (REPO/stage['source']).resolve()
        if not folder.is_relative_to(REPO):
            raise ValueError('Sources must stay within the repository')
        for channel in CHANNELS:
            path = folder/f'{channel}.csv'
            hashes[str(path.relative_to(REPO))] = digest(path)
        motion = load_reference(folder)
        original_frames = len(motion['root'])
        start, stop = stage['source_range']
        stop = original_frames if stop is None else stop
        if not (isinstance(start,int) and isinstance(stop,int) and 0 <= start < stop <= original_frames):
            raise ValueError(f'Invalid source frame interval for {stage["id"]}')
        sliced = {key:value[start:stop].copy() for key,value in motion.items()}
        cleaned, indices, runs = collapse_holds(sliced)
        if len(cleaned['root']) <= 2*count:
            raise ValueError('An overlap would consume an entire action')
        prepared.append(cleaned)
        sources.append({**stage,'original_frames':original_frames,
                        'source_range_half_open':[start,stop],
                        'kept_source_frames':(indices+start).tolist(),
                        'collapsed_holds_relative_to_cut':runs,
                        'prepared_frames':len(cleaned['root']), 'activity':activity(cleaned)})
    motion, seams = prepared[0], []
    for stage, incoming in zip(spec['stages'][1:],prepared[1:]):
        overlap = stage.get('overlap_from_previous_frames', count)
        if not isinstance(overlap,int) or not 3 <= overlap <= min(50,len(incoming['root'])//2):
            raise ValueError('Invalid per-stage overlap')
        begin = len(motion['root'])-overlap
        motion = crossfade(motion,incoming,overlap)
        window = {key:value[begin:begin+overlap].copy() for key,value in motion.items()}
        seams.append({'next_stage_id':stage['id'],'start_frame':begin,
                      'end_frame_exclusive':begin+overlap,'overlap_frames':overlap,
                      'activity':activity(window)})
    _, metrics = describe(motion,kin)
    if metrics['max_joint_limit_violation_rad'] > 1e-5:
        raise ValueError('Strict joint limit check failed; no clipping permitted')
    if metrics['max_joint_velocity_rad_s'] > 15:
        raise ValueError('Reference joint speed exceeds the 15 rad/s screening bound')
    provenance = {'schema_version':1,'source_spec':str(spec_path.relative_to(REPO)),
                  'source_spec_sha256':digest(spec_path),'source_sha256':hashes,
                  'source_stages':sources,'seams':seams,'default_overlap_seconds':count/FPS,
                  'standing_bridge_frames':0,'added_hold_frames':0,
                  'native_active_playback_rate_unchanged':True,
                  'alignment':'XY/yaw align at overlap onset only; source root Z is never translated',
                  'blend':'Quintic moving overlap, shortest-arc quaternion SLERP, FK and velocity rebuild',
                  'activity':activity(motion),'physical_policy_tracking_tested':False,
                  'limitations':'Not a contact optimizer. Existing native gait deceleration may remain; dynamic validation required.'}
    save(output,motion,kin,provenance)
    validation = {'numerical':metrics,'independent_fk':independent_fk(output,kin),
                  'sources_unchanged':all(digest(REPO/path)==value for path,value in hashes.items()),
                  'activity':activity(motion),'dynamic_tracking_tested':False}
    if not validation['sources_unchanged']:
        raise ValueError('Source hashes changed')
    (output/'validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False)+'\n')
    return validation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,default=REPO/'data/bones_seed_continuous_2026-09-16/sequence.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--overlap-seconds',type=float,default=.4)
    args=parser.parse_args()
    print(json.dumps(build(args.spec,args.output,args.overlap_seconds),indent=2))


if __name__=='__main__':
    main()
