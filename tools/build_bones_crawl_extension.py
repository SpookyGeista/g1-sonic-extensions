#!/usr/bin/env python3
"""Append native BONES ground motions without changing the approved walking chain.

Yaw-only Procrustes alignment works for upright AND prone root orientations.
No Euler yaw extraction, world-Z offsets, new holds, or joint-limit clipping.
This builds kinematic references, not a guarantee of physical policy tracking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
import convert_bones_to_sonic_reference as conv
from build_bones_continuous import CHANNELS, activity, independent_fk
from build_bones_walk_video_pack import describe, digest, save
from merge_sonic_reference_motions import load_motion, slerp

REPO = Path(__file__).resolve().parents[1]
FPS = 50


def rotation_matrix(quat):
    """Columns are the world directions of the three local body axes."""
    quat = conv.quat_normalize(np.asarray(quat))
    return np.stack([conv.quat_rotate(quat, np.broadcast_to(axis, quat.shape[:-1]+(3,)))
                     for axis in np.eye(3)], axis=-1)


def yaw_alignment(target, incoming):
    """Optimal world-Z rotation between full orientations, including prone.

    Maximize trace(A.T @ Rz(theta) @ B). Reject the degenerate case rather
    than inventing a heading for incompatible orientations.
    """
    a, b = rotation_matrix(target), rotation_matrix(incoming)
    cosine = np.sum(a[..., 0, :]*b[..., 0, :] + a[..., 1, :]*b[..., 1, :])
    sine = np.sum(a[..., 1, :]*b[..., 0, :] - a[..., 0, :]*b[..., 1, :])
    if np.hypot(cosine, sine) < 1e-6:
        raise ValueError('Cannot resolve world-Z alignment for incompatible poses')
    angle = np.arctan2(sine, cosine)
    return np.array([np.cos(angle/2), 0., 0., np.sin(angle/2)]), float(angle)


def blend_prone_safe(first, incoming, count):
    if not isinstance(count, int) or not 3 <= count < min(len(first['root']),len(incoming['root'])):
        raise ValueError('Overlap must contain at least 3 frames and not consume an action')
    begin = len(first['root'])-count
    q, angle = yaw_alignment(first['quat'][begin], incoming['quat'][0])
    second = {key: value.copy() for key,value in incoming.items()}
    second['root'] = conv.quat_rotate(q,second['root'])
    offset = first['root'][begin]-second['root'][0]
    offset[2] = 0.
    second['root'] += offset
    second['quat'] = conv.quat_mul(q,second['quat'])
    t = np.linspace(0,1,count)
    w = 6*t**5-15*t**4+10*t**3
    result = {}
    for key in first:
        a,b = first[key][begin:],second[key][:count]
        middle = slerp(a,b,w) if key=='quat' else a*(1-w[:,None])+b*w[:,None]
        result[key] = np.concatenate((first[key][:begin],middle,second[key][count:]))
    mismatch = first['joints'][begin:]-second['joints'][:count]
    dot = np.abs(np.sum(first['quat'][begin:]*second['quat'][:count],axis=-1))
    return result, {
        'start_frame':begin,'end_frame_exclusive':begin+count,'overlap_frames':count,
        'world_z_alignment_deg':float(np.rad2deg(angle)),
        'xy_translation_m':offset[:2].tolist(),'z_translation_m':0.,
        'preblend_joint_rms_deg':float(np.rad2deg(np.sqrt(np.mean(mismatch**2)))),
        'preblend_joint_max_deg':float(np.rad2deg(np.abs(mismatch).max())),
        'preblend_root_height_max_difference_m':float(np.abs(first['root'][begin:,2]-second['root'][:count,2]).max()),
        'preblend_orientation_max_difference_deg':float(np.rad2deg(2*np.arccos(np.clip(dot,0,1))).max()),
        'activity':activity({k:v[begin:begin+count] for k,v in result.items()}),
    }


def checked_range(motion, interval):
    start,stop=interval
    if stop is None:stop=len(motion['root'])
    if not isinstance(start,int) or not isinstance(stop,int) or not 0<=start<stop<=len(motion['root']):
        raise ValueError(f'Invalid half-open 50Hz source interval: {interval}')
    return {key:value[start:stop].copy() for key,value in motion.items()},[start,stop]


def reference(path):
    data=load_motion(path)
    return dict(root=data['body_pos'][:,0].copy(),quat=data['body_quat'][:,0].copy(),
                joints=data['joint_pos'][:,conv.MJ_TO_IL].copy())


def repo_path(value):
    path=(REPO/value).resolve()
    if not path.is_relative_to(REPO):raise ValueError('Path must be inside repository')
    return path


def validate_and_save(folder,motion,kin,meta):
    _,metrics=describe(motion,kin)
    # The upright helper's unwrapped Euler yaw has a singularity in prone.
    # Report only the actual upright start/end heading difference instead.
    metrics.pop('net_yaw_deg',None)
    _,heading=yaw_alignment(motion['quat'][-1],motion['quat'][0])
    metrics['upright_endpoint_heading_difference_deg']=float(np.rad2deg(heading))
    if metrics['max_joint_limit_violation_rad']>1e-5:
        raise ValueError('Strict joint limits exceeded; no clipping allowed')
    if metrics['max_joint_velocity_rad_s']>15:
        raise ValueError(f"Reference speed {metrics['max_joint_velocity_rad_s']:.4f}rad/s exceeds existing 15rad/s screening bound")
    tilt=np.rad2deg(np.arccos(np.clip(rotation_matrix(motion['quat'])[:,2,2],-1,1)))
    if not (motion['root'][0,2]>.70 and motion['root'][-1,2]>.70 and max(tilt[[0,-1]])<20):
        raise ValueError('Ground sequence must begin/end upright')
    if not np.any((motion['root'][:,2]<.25)&(tilt>65)):
        raise ValueError('No actual ground phase')
    stored_metrics=save(folder,motion,kin,meta)
    metrics['quaternion_norm_error_after_csv_rounding']=stored_metrics['quaternion_norm_error_after_csv_rounding']
    (folder/'build_metadata.json').write_text(json.dumps({**meta,'validation':metrics},indent=2,allow_nan=False)+'\n')
    result={'numerical':metrics,'independent_fk':independent_fk(folder,kin),
            'activity':activity(motion),'dynamic_tracking_tested':False,
            'root_tilt_max_deg':float(tilt.max()),
            'sources_unchanged':all(digest(REPO/p)==h for p,h in meta['source_sha256'].items())}
    if not result['sources_unchanged']:raise ValueError('Source changed during build')
    (folder/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def build(spec_path,output):
    spec_path=repo_path(spec_path);output=repo_path(output)
    if output.exists():raise ValueError('Choose a NEW output directory; no overwrites')
    spec=json.loads(spec_path.read_text())
    if spec.get('fps')!=FPS or not spec.get('stages'):raise ValueError('Expected nonempty 50Hz spec')
    kin=conv.G1Kinematics(REPO/'gear_sonic_deploy/g1/g1_29dof.xml')
    hashes={str(spec_path.relative_to(REPO)):digest(spec_path)}
    prefix_path=repo_path(spec['prefix']['source'])
    for name in (*[f'{c}.csv' for c in CHANNELS],'build_metadata.json'):
        p=prefix_path/name;hashes[str(p.relative_to(REPO))]=digest(p)
    original=reference(prefix_path)
    prefix,prefix_range=checked_range(original,spec['prefix']['source_range'])
    if prefix_range[0]!=0:raise ValueError('Original prefix must start at frame0')
    old=json.loads((prefix_path/'build_metadata.json').read_text())
    starts=[0]+[seam['start_frame'] for seam in old['seams']]
    labels=['Gehen','Deckung links','Linkskurve','Gehen','Deckung rechts','Rechtskurve','Gehen']
    timeline=[dict(id=stage['id'],label=labels[i],start_frame=start,
                   end_frame_exclusive=(starts+[len(prefix['root'])])[i+1])
              for i,(stage,start) in enumerate(zip(old['source_stages'],starts))]
    pieces=[];sources=[]
    for stage in spec['stages']:
        path=repo_path(stage['source']);hashes[str(path.relative_to(REPO))]=digest(path)
        root,quat,joints=conv.read_bones_csv(path,kin)
        root,quat,joints=conv.resample(root,quat,joints,120,FPS)
        piece,interval=checked_range(dict(root=root,quat=quat,joints=joints),stage['source_range'])
        pieces.append(piece)
        sources.append({**stage,'resampled_range_half_open':interval,
                        'prepared_frames':len(piece['root']),'activity':activity(piece)})
    common={'schema_version':1,'fps':FPS,'source_fps':120,
            'source_fps_basis':'Existing repository BONES convention; no FPS field in CSV',
            'source_spec':str(spec_path.relative_to(REPO)),'source_sha256':hashes,
            'source_stages':sources,'prefix':{**spec['prefix'],'source_range_half_open':prefix_range},
            'alignment':'Full-orientation world-Z Procrustes plus XY translation; no Euler yaw; root Z preserved',
            'blend':'Moving quintic overlap and shortest-arc quaternion SLERP; FK/velocities rebuilt',
            'added_hold_frames':0,'standing_bridge_frames':0,'native_active_playback_rate_unchanged':True,
            'physical_policy_tracking_tested':False,
            'limitations':'Kinematic references only; no contact optimizer, no claim of SONIC crawl capability'}
    complete=prefix;seams=[]
    for stage,piece in zip(spec['stages'],pieces):
        complete,seam=blend_prone_safe(complete,piece,stage['overlap_from_previous_frames'])
        seam['next_stage_id']=stage['id'];seams.append(seam)
        timeline[-1]['end_frame_exclusive']=seam['start_frame']
        timeline.append(dict(id=stage['id'],label=stage['label'],start_frame=seam['start_frame'],
                             end_frame_exclusive=len(complete['root'])))
    # Prefix poses outside the first new overlap must remain exactly unchanged.
    unchanged=seams[0]['start_frame']
    for key in original:np.testing.assert_array_equal(original[key][:unchanged],complete[key][:unchanged])
    crawl={k:v.copy() for k,v in pieces[0].items()};crawl_seams=[]
    crawl_timeline=[dict(id=spec['stages'][0]['id'],label=spec['stages'][0]['label'],
                         start_frame=0,end_frame_exclusive=len(crawl['root']))]
    for stage,piece in zip(spec['stages'][1:],pieces[1:]):
        crawl,seam=blend_prone_safe(crawl,piece,stage['overlap_from_previous_frames'])
        seam['next_stage_id']=stage['id'];crawl_seams.append(seam)
        crawl_timeline[-1]['end_frame_exclusive']=seam['start_frame']
        crawl_timeline.append(dict(id=stage['id'],label=stage['label'],start_frame=seam['start_frame'],end_frame_exclusive=len(crawl['root'])))
    # Normalize crawl-only in upright initial pose, using the same full-frame fit.
    q,_=yaw_alignment(np.array([1.,0,0,0]),crawl['quat'][0])
    origin=crawl['root'][0].copy();origin[2]=0
    crawl['root']=conv.quat_rotate(q,crawl['root']-origin)
    crawl['quat']=conv.quat_mul(q,crawl['quat'])
    results={}
    for name,motion,tl,sm in [('01_complete_chain',complete,timeline,seams),('02_crawl_only',crawl,crawl_timeline,crawl_seams)]:
        results[name]=validate_and_save(output/name,motion,kin,{**common,'timeline':tl,'seams':sm,
            'unchanged_approved_prefix_frames':unchanged if name=='01_complete_chain' else 0})
    (output/'build_report.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(build(args.spec,args.output),indent=2))
