#!/usr/bin/env python3
"""Shorten reviewed quiet intervals without changing the active route/poses.

50 Hz output; monotonic quintic time warp with unit endpoint slope. No spatial
alignment, added holds, controller changes or joint clipping. Outputs are new.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import convert_bones_to_sonic_reference as conv
from build_bones_crawl_extension import reference, validate_and_save, REPO
from build_bones_continuous import CHANNELS, activity
from build_bones_walk_video_pack import digest
from merge_sonic_reference_motions import slerp


def time_map(frames, windows):
    pieces=[]; cursor=0
    for window in windows:
        lo,hi=window['source_frames_inclusive']
        edges=window['output_intervals']
        if not all(type(v) is int for v in (lo,hi,edges)) or not cursor<=lo<hi<frames or not 3<=edges<hi-lo:
            raise ValueError('Ordered, disjoint source windows and 3..N-1 output intervals required')
        pieces.append(np.arange(cursor,lo,dtype=float))
        t=np.linspace(0,1,edges+1)
        pieces.append(lo+edges*t+(hi-lo-edges)*(6*t**5-15*t**4+10*t**3))
        cursor=hi+1
    pieces.append(np.arange(cursor,frames,dtype=float))
    result=np.concatenate(pieces)
    if result[0]!=0 or result[-1]!=frames-1 or np.any(np.diff(result)<=0):
        raise ValueError('Time warp must preserve endpoints and strictly increase')
    return result


def retime(motion, source_frames):
    lo=np.floor(source_frames).astype(int)
    hi=np.minimum(lo+1,len(motion['root'])-1)
    weight=source_frames-lo
    result={}
    for key,value in motion.items():
        result[key]=slerp(value[lo],value[hi],weight) if key=='quat' else value[lo]*(1-weight[:,None])+value[hi]*weight[:,None]
    return result


def remap_timeline(timeline, mapping):
    result=[]
    for stage in timeline:
        result.append({**stage,
            'source_start_frame':stage['start_frame'],
            'source_end_frame_exclusive':stage['end_frame_exclusive'],
            'start_frame':int(np.searchsorted(mapping,stage['start_frame'])),
            'end_frame_exclusive':int(np.searchsorted(mapping,stage['end_frame_exclusive']))})
    return result


def build(spec_path, output):
    spec_path=spec_path.resolve(); output=output.resolve()
    if not output.is_relative_to(REPO) or output.exists():
        raise ValueError('Choose a NEW output inside repository')
    spec=json.loads(spec_path.read_text())
    source=(REPO/spec['source']).resolve()
    if not source.is_relative_to(REPO) or spec['fps']!=50:
        raise ValueError('Repository source and 50 Hz required')
    motion=reference(source)
    if len(motion['root'])!=spec['source_frames']:
        raise ValueError('Source frame count changed; review quiet windows again')
    source_meta=json.loads((source/'build_metadata.json').read_text())
    mapping=time_map(len(motion['root']),spec['windows'])
    result=retime(motion,mapping)
    # Active poses outside reviewed quiet windows are identical (quat up to sign).
    exact=mapping==np.round(mapping)
    for key in ('root','joints'):
        np.testing.assert_array_equal(result[key][exact],motion[key][mapping[exact].astype(int)])
    files=[spec_path]+[source/f'{channel}.csv' for channel in CHANNELS]+[source/'build_metadata.json',source/'metadata.txt']
    hashes={str(p.relative_to(REPO)):digest(p) for p in files}
    diagnostics=[]
    for w in spec['windows']:
        lo,hi=w['source_frames_inclusive']
        q=motion['joints'][lo:hi+1]; root=motion['root'][lo:hi+1]
        diagnostics.append({**w,'source_seconds':(hi-lo)/50,'new_seconds':w['output_intervals']/50,
            'root_endpoint_displacement_m':float(np.linalg.norm(root[-1]-root[0])),
            'largest_joint_range_rad':float(np.ptp(q,axis=0).max())})
    meta={'schema_version':1,'fps':50,'source_sha256':hashes,'source':str(source.relative_to(REPO)),
        'method':'Unit-endpoint-slope quintic time warp inside reviewed quiet windows; linear root/joints, quaternion SLERP; FK and all velocities rebuilt',
        'timeline':remap_timeline(source_meta['timeline'],mapping),'quiet_windows':diagnostics,
        'removed_frames':len(motion['root'])-len(mapping),
        'removed_seconds':(len(motion['root'])-len(mapping))/50,
        'source_activity':activity(motion),'source_frame_mapping':'source_frame_mapping.csv',
        'spatial_path_relocated':False,'active_integer_poses_preserved':True,
        'source_timing_inside_quiet_windows_changed':True,'physical_policy_tracking_tested':False}
    kin=conv.G1Kinematics(REPO/'gear_sonic_deploy/g1/g1_29dof.xml')
    report=validate_and_save(output,result,kin,meta)
    np.savetxt(output/'source_frame_mapping.csv',np.column_stack((np.arange(len(mapping)),mapping)),
               delimiter=',',header='output_frame,source_fractional_frame',comments='',fmt=['%d','%.12f'])
    return {'frames':len(mapping),'duration_seconds':(len(mapping)-1)/50,
            'removed_seconds':meta['removed_seconds'],'validation':report}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.spec,args.output),indent=2,allow_nan=False))
