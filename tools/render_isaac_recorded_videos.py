#!/usr/bin/env python3
"""Re-render one recorded PhysX run from three cameras, plus a chronological cut.

No controller, DDS or physics rollout. Root/joints AND camera poses come from
the recorded run, with time-aligned interpolation. Source recordings untouched.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO=Path(__file__).resolve().parents[1]
FFMPEG=Path('/home/loya2/miniconda3/envs/unitree_sim_env/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
FILENAMES={'top':'01_Draufsicht.mp4','first':'02_G1_Sicht.mp4','chase':'03_Verfolger.mp4','mix':'04_Kameramix.mp4'}


def reject_active_simulation():
    """Never contend with another realtime SONIC/Isaac run; do not stop it."""
    names={'g1_deploy_onnx_ref','run_isaac.py','run_sim_loop.py','render_isaac_recorded_videos.py'}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:argv=[os.fsdecode(x) for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if any(Path(token).name in names for token in argv[:4]):
            raise RuntimeError(f'Existing simulator/controller/renderer PID {proc.name}; finish it before rendering videos')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interp(times,values,target_times):
    values=np.asarray(values)
    return np.column_stack([np.interp(target_times,times,values[:,i]) for i in range(values.shape[1])])


def camera_rotations(records,name):
    eye=np.array([r[name]['eye'] for r in records],dtype=float)
    target=np.array([r[name]['target'] for r in records],dtype=float)
    up=np.array([r[name]['up'] for r in records],dtype=float)
    backward=eye-target;backward/=np.linalg.norm(backward,axis=1)[:,None]
    right=np.cross(up,backward);right/=np.linalg.norm(right,axis=1)[:,None]
    corrected_up=np.cross(backward,right)
    return Rotation.from_matrix(np.stack((right,corrected_up,backward),axis=-1))


def load_recording(trial,fps=25,frame_limit=None):
    summary=json.loads((trial/'summary.json').read_text())
    if not summary['all_requested_stages_completed'] or summary['observed_fall']:
        raise ValueError('Use a completed non-fallen run; do not hide failed execution')
    if summary['source_or_runtime_hash_changes'] or not summary['timing']['passed']:
        raise ValueError('Source hashes or physical timing were invalid')
    with (trial/'controller/motion_playing.csv').open() as handle:
        playing=[r for r in csv.DictReader(handle) if float(r['playing_0'])>.5]
    indexes=np.array([int(r['index']) for r in playing])
    if len(playing)<2 or np.any(np.diff(indexes)!=1):raise ValueError('Noncontiguous playback')
    start,end=(float(playing[i]['time_monotonic_ms'])/1000 for i in (0,-1))
    duration=end-start
    # Include the final pose; its last display interval gives a <=1/fps extension.
    count=int(math.floor(duration*fps+1e-5))+1
    if frame_limit:count=min(count,frame_limit)
    wall_times=start+np.minimum(np.arange(count)/fps,duration)
    with np.load(trial/'sim/trace.npz',allow_pickle=False) as saved:
        data=saved['data'];cols=saved['columns'].tolist()
    ix={n:i for i,n in enumerate(cols)}
    wall=data[:,ix['wall_monotonic']];sim=data[:,ix['sim_time']]
    if np.any(np.diff(wall)<=0) or np.any(np.diff(sim)<=0):raise ValueError('Nonmonotonic trace')
    if wall_times[0]<wall[0] or wall_times[-1]>wall[-1]:raise ValueError('Trace does not cover full video')
    sample_times=np.interp(wall_times,wall,sim)
    raw_root=data[:,[ix[f'root_{i}'] for i in range(7)]]
    root=np.column_stack((interp(sim,raw_root[:,:3],sample_times),
        Slerp(sim,Rotation.from_quat(raw_root[:,3:7],scalar_first=True))(sample_times).as_quat(scalar_first=True)))
    joints=interp(sim,data[:,[ix[f'q_{i}'] for i in range(29)]],sample_times)
    records=json.loads((trial/'sim/cameras.json').read_text())
    cam_times=np.array([r['sim_time'] for r in records])
    if sample_times[0]<cam_times[0] or sample_times[-1]>cam_times[-1]:raise ValueError('Camera trace too short')
    cameras={}
    for name in ('first','chase'):
        eye=interp(cam_times,np.array([r[name]['eye'] for r in records]),sample_times)
        rotation=Slerp(cam_times,camera_rotations(records,name))(sample_times).as_matrix()
        cameras[name]={'eye':eye,'target':eye-rotation[:,:,2],'up':rotation[:,:,1]}
    report={'source_playback_start_monotonic':start,'source_playback_end_monotonic':end,
        'source_playback_duration_s':duration,'source_controller_rows':len(playing),
        'video_frames':count,'video_fps':fps,'encoded_duration_s':count/fps,
        'first_to_last_sample_s':(count-1)/fps,
        'sample_sim_time_range_s':[float(sample_times[0]),float(sample_times[-1])],
        'alignment':'Controller monotonic time -> measured trace sim time -> pose and camera at the same sim time; no extra pre/post-step offset',
        'root_position_and_joints':'linear interpolation of measured 200 Hz states',
        'root_and_camera_orientation':'shortest-arc quaternion SLERP',
        'ground_tracking_passed':summary['ground_tracking']['passed'],
        'source_overall_passed':summary['passed']}
    return root,joints,cameras,report


def validate_edit(edit,duration):
    if not edit or edit[0]['start_s']!=0:raise ValueError('Mix must start at zero')
    for i,item in enumerate(edit):
        if item['camera'] not in ('top','first','chase'):raise ValueError('Unknown mix camera')
        if item['start_s']<0 or item['end_s']<=item['start_s']:raise ValueError('Invalid edit interval')
        if i and abs(edit[i-1]['end_s']-item['start_s'])>1e-9:raise ValueError('Gap/overlap in mix')
    if edit[-1]['end_s']<duration-1e-6:raise ValueError('Mix must cover complete video')


def chosen_view(edit,seconds):
    return next(item['camera'] for item in edit if item['start_s']<=seconds<item['end_s'])


class Encoder:
    def __init__(self,path,width,height,fps):
        self.path=path;self.log=path.with_suffix('.ffmpeg.log').open('wb')
        self.command=[str(FFMPEG),'-hide_banner','-loglevel','warning','-n',
            '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
            '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast',
            '-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',
            '-metadata','comment=Re-rendered measured Isaac PhysX run; no new physics rollout; source ground tracking not fully passed',str(path)]
        self.process=subprocess.Popen(self.command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.log)
        self.frames=0;self.closed=False

    def write(self,pixels):
        self.process.stdin.write(np.ascontiguousarray(pixels[...,:3],dtype=np.uint8).tobytes())
        self.frames+=1

    def close(self):
        if self.closed:return
        self.closed=True
        self.process.stdin.close()
        try:code=self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.terminate();self.process.wait(timeout=10);raise
        finally:self.log.close()
        if code:raise RuntimeError(f'Encoder failed ({code}): {self.path}')


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-dir',type=Path,required=True)
    p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--edit',type=Path,required=True)
    p.add_argument('--width',type=int,default=1920);p.add_argument('--height',type=int,default=1080)
    p.add_argument('--fps',type=int,default=25)
    p.add_argument('--frame-limit',type=int,help='Short smoke export only; not full videos')
    p.add_argument('--render-updates',type=int,default=3,help='Render-only settling updates per sampled pose')
    AppLauncher.add_app_launcher_args(p);args=p.parse_args()
    args.headless=True;args.enable_cameras=True
    for name in ('scene_dir','trial','output','edit'):setattr(args,name,getattr(args,name).resolve())
    if not args.output.is_relative_to(REPO) or args.output.exists():p.error('NEW output directory inside repository required')
    if args.width%2 or args.height%2 or not 640<=args.width<=3840 or not 360<=args.height<=2160:p.error('Even video dimensions 640..3840 x 360..2160 required')
    if args.fps!=25 or args.render_updates<2 or (args.frame_limit is not None and args.frame_limit<2):p.error('25 fps, >=2 rendering updates and frames required')
    if not FFMPEG.is_file():p.error('Missing known local FFmpeg')
    reject_active_simulation()
    os.chdir(REPO);sys.path.insert(0,str(args.scene_dir))
    from bridge import BODY_JOINT_NAMES
    root,joints,cameras,report=load_recording(args.trial,args.fps,args.frame_limit)
    layout=json.loads((args.scene_dir/'layout.json').read_text())
    recorded=json.loads((args.trial/'sim/metadata.json').read_text())
    edit=json.loads(args.edit.read_text())['segments']
    validate_edit(edit,report['encoded_duration_s'])
    files=[Path(__file__),args.edit,args.scene_dir/'layout.json',args.scene_dir/'scene.py',args.scene_dir/'cameras.py',
        args.trial/'sim/trace.npz',args.trial/'sim/cameras.json',args.trial/'sim/metadata.json',
        args.trial/'controller/motion_playing.csv',args.trial/'summary.json']
    hashes={str(path):digest(path) for path in files}
    args.output.mkdir(parents=True,exist_ok=False)
    report.update(kind='Re-render of recorded Isaac/PhysX run, not a new dynamics validation',
        source_sha256=hashes,no_dds=True,physics_steps_advanced=0,resolution=[args.width,args.height],
        mix_edit=edit,render_updates_per_frame=args.render_updates,status='started',videos={})
    (args.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    app=AppLauncher(args).app;encoders={};products={};annotators={}
    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        import omni.replicator.core as rep
        from PIL import Image
        from scene import build_scene
        from cameras import CameraRig
        torch.set_num_threads(1);rep.orchestrator.set_capture_on_play(False)
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=args.device))
        scene=build_scene(hall=False,course=True)
        if scene['collision_boxes']!=recorded['scene']['collision_boxes']:
            raise ValueError('Scene geometry differs from measured recording')
        spec=importlib.util.spec_from_file_location('sonic_training_robot_cfg',REPO/'gear_sonic/envs/manager_env/robots/g1.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cfg=module.G1_CYLINDER_MODEL_12_DEX_CFG.copy();cfg.prim_path='/World/G1'
        cfg.spawn.asset_path=str(REPO/cfg.spawn.asset_path)
        cfg.spawn.usd_dir=str(args.scene_dir/'robot_cache');cfg.spawn.usd_file_name='sonic_training_g1.usd'
        for actuator in cfg.actuators.values():actuator.stiffness=0.;actuator.damping=0.
        robot=Articulation(cfg);rig=CameraRig(layout,scene['collision_boxes'],'top')
        sim.reset();sim.set_setting('/app/runLoops/main/rateLimitEnabled',False)
        indices=[list(robot.joint_names).index(n) for n in BODY_JOINT_NAMES]
        step_index=int(sim.current_time_step_index)
        for name,path in rig.paths.items():
            products[name]=rep.create.render_product(path,(args.width,args.height))
            annotators[name]=rep.AnnotatorRegistry.get_annotator('rgb')
            annotators[name].attach(products[name])
        for name,filename in FILENAMES.items():encoders[name]=Encoder(args.output/filename,args.width,args.height,args.fps)
        snapshots=set([0,len(root)-1]+[min(len(root)-1,round(t*args.fps)) for t in (4,8,13,17,22,28,34,38,44)])
        started=time.monotonic();last_progress=started;frame_hashes=[]
        for i in range(len(root)):
            pose=torch.tensor(root[i],dtype=torch.float32,device=args.device).reshape(1,7)
            q=robot.data.default_joint_pos.clone()
            q[:,indices]=torch.tensor(joints[i],dtype=torch.float32,device=args.device)
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(torch.zeros((1,6),device=args.device))
            robot.write_joint_state_to_sim(q,torch.zeros_like(q));sim.forward()
            for name in ('first','chase'):rig.set_pose(name,{k:v[i] for k,v in cameras[name].items()})
            for _ in range(24 if i==0 else args.render_updates):sim.render()
            if int(sim.current_time_step_index)!=step_index:raise RuntimeError('Physics advanced while reconstructing recorded video')
            frames={name:np.asarray(annotator.get_data())[...,:3].copy() for name,annotator in annotators.items()}
            if any(v.shape!=(args.height,args.width,3) or v.dtype!=np.uint8 for v in frames.values()):
                raise ValueError('Unexpected rendered RGB shape/type')
            chosen=chosen_view(edit,i/args.fps)
            for name,pixels in frames.items():encoders[name].write(pixels)
            encoders['mix'].write(frames[chosen])
            frame_hashes.append({name:hashlib.sha256(pixels.tobytes()).hexdigest() for name,pixels in frames.items()})
            if i in snapshots:
                for name,pixels in frames.items():Image.fromarray(pixels).save(args.output/f'preview_{i:04d}_{name}.jpg',quality=90)
            now=time.monotonic()
            if now-last_progress>=10 or i==len(root)-1:
                print('VIDEO_PROGRESS '+json.dumps({'frame':i+1,'total':len(root),'clip_seconds':round(i/args.fps,2),'elapsed_s':round(now-started,1)}),flush=True)
                last_progress=now
        for encoder in encoders.values():encoder.close()
        if any(digest(path)!=value for path,value in hashes.items()):raise RuntimeError('Source changed during export')
        report.update(status='rendered',source_files_unchanged=True,elapsed_render_seconds=time.monotonic()-started,
            frame_rgb_hashes=frame_hashes,
            videos={name:{'file':FILENAMES[name],'frames':encoder.frames,'bytes':encoder.path.stat().st_size,
                          'sha256':digest(encoder.path)} for name,encoder in encoders.items()})
        (args.output/'provenance.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('VIDEOS_RENDERED '+json.dumps(report['videos']),flush=True)
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        (args.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
        raise
    finally:
        for encoder in encoders.values():
            try:encoder.close()
            except Exception:pass
        # No further render/update follows. This Kit version can invalidate
        # shared annotator nodes while detaching multiple camera products.
        # Let the standalone immediate shutdown release them together instead.
        app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=='__main__':main()
