#!/usr/bin/env python3
"""Build isolated BONES walking modules and a configurable offline reference chain.

No deployment processes or robot communication. Reuses existing G1 parser/FK.
Original CSVs, ready motions and existing setup files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

import convert_bones_to_sonic_reference as conv
from build_bones_crouch_pair import load, crossfade, add_holds, normalize_origin
from merge_sonic_reference_motions import load_motion, calculate_velocities, slerp, yaw_quat

REPO = Path(__file__).resolve().parents[1]
FPS = 50


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference(path):
    data = load_motion(path)
    root, quat = normalize_origin(data['body_pos'][:, 0], data['body_quat'][:, 0])
    return dict(root=root, quat=quat, joints=data['joint_pos'][:, conv.MJ_TO_IL])


def join_standing(first, second, seconds):
    """Yaw/XY-align and bridge standing endpoints BEFORE FK/policy, not motor outputs."""
    count = round(seconds * FPS)
    if count < 2:
        raise ValueError('Standing bridges require at least two frames')
    second = {key: value.copy() for key, value in second.items()}
    rotation = conv.quat_mul(yaw_quat(first['quat'][-1]),
                             conv.quat_conjugate(yaw_quat(second['quat'][0])))
    second['root'] = conv.quat_rotate(rotation, second['root'])
    offset = first['root'][-1] - second['root'][0]
    offset[2] = 0  # Never translate the recorded ground plane vertically.
    second['root'] += offset
    second['quat'] = conv.quat_mul(rotation, second['quat'])
    t = np.linspace(0, 1, count + 2)[1:-1]
    weight = 6*t**5 - 15*t**4 + 10*t**3
    result = {}
    for key in first:
        a = np.repeat(first[key][-1:], count, axis=0)
        b = np.repeat(second[key][:1], count, axis=0)
        bridge = slerp(a, b, weight) if key == 'quat' else a*(1-weight[:, None])+b*weight[:, None]
        result[key] = np.concatenate((first[key], bridge, second[key]))
    seam = {
        'start_frame': len(first['root']) - 1, 'bridge_frames': count,
        'joint_endpoint_rms_deg': float(np.rad2deg(np.sqrt(np.mean((first['joints'][-1]-second['joints'][0])**2)))),
        'joint_endpoint_max_deg': float(np.rad2deg(abs(first['joints'][-1]-second['joints'][0]).max())),
        'root_height_difference_m': float(second['root'][0, 2] - first['root'][-1, 2]),
        'note': 'Reference interpolation, not a contact-constrained trajectory optimizer.',
    }
    return result, seam


def describe(motion, kin):
    root, quat, joints = (motion[k] for k in ('root', 'quat', 'joints'))
    if len(root) < 3 or joints.shape != (len(root), 29):
        raise ValueError('Expected at least 3 frames with 29 joints')
    if not all(np.isfinite(value).all() for value in motion.values()):
        raise ValueError('Nonfinite motion data')
    if np.max(abs(np.linalg.norm(quat, axis=-1)-1)) > 1e-5:
        raise ValueError('Root quaternions are not normalized')
    limits = np.asarray(kin.joint_limits)
    violation = float(np.maximum(np.maximum(limits[:, 0]-joints, joints-limits[:, 1]), 0).max())
    if violation > np.deg2rad(1):
        raise ValueError(f'Joint limit violation: {violation} rad')
    w,x,y,z = quat.T
    yaw = np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    body_pos, body_quat = kin.forward(root, quat, joints)
    joint_pos = np.empty_like(joints)
    joint_pos[:, conv.MJ_TO_IL] = joints
    joint_vel, body_lin_vel, body_ang_vel = calculate_velocities(joint_pos, body_pos, body_quat)
    arrays = dict(joint_pos=joint_pos, joint_vel=joint_vel, body_pos=body_pos,
                  body_quat=body_quat, body_lin_vel=body_lin_vel, body_ang_vel=body_ang_vel)
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite generated channels')
    metrics = {
        'frames': len(root), 'fps': FPS, 'duration_seconds': (len(root)-1)/FPS,
        'root_height_start_min_end_m': [float(root[0,2]), float(root[:,2].min()), float(root[-1,2])],
        'net_displacement_m': float(np.linalg.norm(root[-1,:2]-root[0,:2])),
        'xy_path_length_m': float(np.linalg.norm(np.diff(root[:,:2],axis=0),axis=1).sum()),
        'net_yaw_deg': float(np.rad2deg(yaw[-1]-yaw[0])),
        'knee_start_end_deg': np.rad2deg(joints[[0,-1]][:, [3,9]]).tolist(),
        'max_joint_velocity_rad_s': float(abs(joint_vel).max()),
        'max_joint_step_rad': float(abs(np.diff(joints,axis=0)).max()),
        'max_joint_limit_violation_rad': violation, 'finite': True, 'joint_count': 29,
        'reference_only_validation': True,
    }
    return arrays, metrics


def save(folder, motion, kin, provenance):
    arrays, metrics = describe(motion, kin)
    folder.mkdir(parents=True, exist_ok=False)
    for key, value in arrays.items():
        if key in ('joint_pos','joint_vel'):
            header = [f'{key}_{i}' for i in range(29)]
        else:
            header = [f'body_{i}_{axis}' for i in range(14) for axis in ('wxyz' if key=='body_quat' else 'xyz')]
        conv.save_csv(folder/f'{key}.csv', value, header)
    (folder/'metadata.txt').write_text(
        f'Metadata for: {folder.name}\n==============================\n\nBody part indexes:\n'
        f'{np.array2string(conv.BODY_INDEXES_IL)}\n\nTotal timesteps: {len(motion["root"])}\n')
    # load_motion normalizes quaternions; inspect the actual rounded CSV instead.
    rounded_quat = np.loadtxt(folder/'body_quat.csv',delimiter=',',skiprows=1).reshape(-1,14,4)
    error = float(abs(np.linalg.norm(rounded_quat,axis=-1)-1).max())
    if error > 1e-5:
        raise ValueError('Quaternion precision lost in CSV output')
    metrics['quaternion_norm_error_after_csv_rounding'] = error
    (folder/'build_metadata.json').write_text(json.dumps({**provenance,'validation':metrics},indent=2)+'\n')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New directory only')
    parser.add_argument('--sequence',nargs='+',help='Override example chain using module numbers or full IDs; repeats allowed')
    parser.add_argument('--chain-name',default='01_custom_chain')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    spec = json.loads(args.spec.read_text())
    kin = conv.G1Kinematics(REPO/'gear_sonic_deploy/g1/g1_29dof.xml')
    ids = [module['id'] for module in spec['modules']]
    if len(set(ids)) != len(ids) or any(Path(name).name != name or name in ('.','..') for name in ids):
        parser.error('Module IDs must be unique simple directory names')
    if args.sequence:
        sequence = []
        for token in args.sequence:
            matches = [name for name in ids if name == token or
                       (token.isdigit() and name.split('_',1)[0] == f'{int(token):02d}')]
            if len(matches) != 1:
                parser.error(f'Unknown or ambiguous module: {token}')
            sequence.append(matches[0])
        if Path(args.chain_name).name != args.chain_name or args.chain_name in ('.','..'):
            parser.error('Chain name must be a simple folder name')
        spec['chains'] = [{'id':args.chain_name,'sequence':sequence,'bridge_seconds':1}]
    for module in spec['modules']:
        sources = [module['copy_reference']] if 'copy_reference' in module else [part['source'] for part in module['parts']]
        for source in sources:
            if not (REPO/source).exists():
                parser.error(f'Missing source: {source}')
    args.output.mkdir(parents=True)
    motions, manifest = {}, {}
    for module in spec['modules']:
        name = module['id']
        dest = args.output/'ready'/name
        if 'copy_reference' in module:
            source = REPO/module['copy_reference']
            shutil.copytree(source,dest)
            hashes = {p.name:digest(p) for p in source.iterdir() if p.is_file()}
            if any(digest(dest/file)!=sha for file,sha in hashes.items()):
                raise RuntimeError('Original copy hash mismatch')
            motions[name] = load_reference(source)
            _, metrics = describe(motions[name],kin)
            manifest[name] = {'copy_source':str(source),'copied_file_sha256':hashes,'validation':metrics}
            continue
        pieces, sources = [], []
        for part in module['parts']:
            source = REPO/part['source']
            motion = load(source, kin)
            start = part.get('start_frame',0)
            stop = part.get('stop_frame',len(motion['root']))
            if not 0 <= start < stop <= len(motion['root']):
                raise ValueError(f'Invalid 50Hz crop: {part}')
            pieces.append({key:value[start:stop].copy() for key,value in motion.items()})
            sources.append({'source':part['source'],'sha256':digest(source),
                            'resampled_range_half_open':[start,stop]})
        motion = pieces[0]
        overlap = module.get('overlap_frames',25)
        for piece in pieces[1:]:
            if not 2 <= overlap < min(len(piece['root']),len(motion['root'])):
                raise ValueError('Invalid walking overlap count')
            motion = crossfade(motion,piece,overlap)
        motion = add_holds(motion,round(module.get('hold_seconds',1)*FPS))
        motions[name] = motion
        provenance = {**module,'sources':sources,'source_fps':120,'target_fps':FPS,
                      'source_fps_basis':'Existing repository BONES converter; CSV has frame index, no FPS field',
                      'conversion':'Existing cm→m; deg→rad; extrinsic XYZ→wxyz; 120→50Hz; MuJoCo→IsaacLab',
                      'body_transforms':'Recomputed G1 FK after reference blending; all velocities recomputed',
                      'alignment':'Initial XY/yaw zero; overlap yaw/XY aligned; root Z never offset'}
        manifest[name] = {'validation':save(dest,motion,kin,provenance),'sources':sources}
    for chain in spec.get('chains',[]):
        order = chain['sequence']
        if not order or any(name not in motions for name in order):
            raise ValueError('Chain references an unknown module')
        if Path(chain['id']).name != chain['id'] or chain['id'] in ('.','..'):
            raise ValueError('Chain ID must be a simple directory name')
        motion, seams = motions[order[0]], []
        for name in order[1:]:
            motion, seam = join_standing(motion,motions[name],chain.get('bridge_seconds',1))
            seams.append({'next_module':name,**seam})
        metrics = save(args.output/'chains'/chain['id'],motion,kin,{**chain,'seams':seams,
            'body_transforms':'Recomputed G1 FK; source originals unmodified',
            'warning':'Cross-actor standing bridges are kinematic interpolations; physical tracking must be tested separately'})
        manifest[chain['id']] = {'sequence':order,'seams':seams,'validation':metrics}
    (args.output/'pack_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({name:item['validation'] for name,item in manifest.items()},indent=2))


if __name__ == '__main__':
    main()
