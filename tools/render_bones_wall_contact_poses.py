#!/usr/bin/env python3
"""Render measured cover poses in their exact scene, without DDS or a new rollout.

These are reconstructions from recorded joint/root states, not live screenshots
or independent dynamics/contact validation. No physics steps are advanced.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera_views(layout, name, position, doorway_views=False):
    """Same-side overview plus optional view through the adjacent doorway."""
    import numpy as np
    pos = np.asarray(position, dtype=float)
    result = [(name, pos+[-1.8,-2.4,1.4], pos+[.12,.12,.2])]
    if doorway_views:
        portals = {p['name']:p for p in layout['portals']}
        if name == 'deckung_links':
            p = portals['CoverDoor01']; x,y = p['center_xy']
            eye = [x+.9, y+p['clear_width']/2-.6, 1.05]
        elif name == 'deckung_rechts':
            p = portals['CoverDoor02']; x,y = p['center_xy']
            eye = [x+p['clear_width']/2-.6, y+.9, 1.05]
        else:
            raise ValueError(f'Unknown cover pose: {name}')
        result.append((name+'_durchgang', np.asarray(eye), pos+[0,0,.2]))
    return result


def main():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-dir', type=Path, required=True)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--doorway-views', action='store_true', help='Also render the view from the other side of each cover doorway')
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.headless = True
    args.enable_cameras = True
    args.scene_dir = args.scene_dir.resolve()
    args.trial = args.trial.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError('Choose a new reconstruction output')
    import numpy as np
    sys.path.insert(0, str(args.scene_dir))
    from bridge import BODY_JOINT_NAMES
    trace_path = args.trial/'sim/trace.npz'
    scene_path = args.trial/'sim/metadata.json'
    layout_path = args.scene_dir/'layout.json'
    layout = json.loads(layout_path.read_text())
    hashes = {str(path): digest(path) for path in (trace_path, scene_path, layout_path)}
    with np.load(trace_path, allow_pickle=False) as saved:
        data = saved['data']; columns = saved['columns'].tolist()
    ix = {n:i for i,n in enumerate(columns)}
    rows = []
    for name, frame in [('deckung_links', 300), ('deckung_rechts', 965)]:
        selected = np.flatnonzero((data[:,ix['guard_reference_frame']]==frame) & (data[:,ix['support']]==0))
        if not len(selected):
            raise ValueError(f'Requested measured cover frame {frame} absent')
        rows.append((name, frame, int(selected[len(selected)//2])))
    recorded_scene = json.loads(scene_path.read_text())['scene']
    args.output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(args).app
    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        import omni.replicator.core as rep
        from PIL import Image, ImageDraw
        from scene import build_scene
        rep.orchestrator.set_capture_on_play(False)
        torch.set_num_threads(1)
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device=args.device))
        scene = build_scene(hall=False, course=True)
        for actual, expected in zip(scene['collision_boxes'], recorded_scene['collision_boxes']):
            if actual['prim'] != expected['prim'] or not all(np.allclose(actual[k], expected[k], atol=1e-7, rtol=0) for k in ('aabb_min','aabb_max')):
                raise ValueError('Current scene differs from recorded trial')
        if len(scene['collision_boxes']) != len(recorded_scene['collision_boxes']):
            raise ValueError('Current obstacle count differs from recorded trial')
        spec = importlib.util.spec_from_file_location('sonic_training_robot_cfg', REPO/'gear_sonic/envs/manager_env/robots/g1.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        cfg = module.G1_CYLINDER_MODEL_12_DEX_CFG.copy()
        cfg.prim_path = '/World/G1'
        cfg.spawn.asset_path = str(REPO/cfg.spawn.asset_path)
        cfg.spawn.usd_dir = str(args.scene_dir/'robot_cache')
        cfg.spawn.usd_file_name = 'sonic_training_g1.usd'
        for actuator in cfg.actuators.values():
            actuator.stiffness = 0.; actuator.damping = 0.
        robot = Articulation(cfg)
        sim.reset()
        indices = [list(robot.joint_names).index(n) for n in BODY_JOINT_NAMES]
        product = rep.create.render_product('/OmniverseKit_Persp', (1440,1080))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach(product)
        images = []
        step_index = sim.current_time_step_index
        for name, frame, row_index in rows:
            row = data[row_index]
            root = torch.tensor(row[[ix[f'root_{i}'] for i in range(7)]], dtype=torch.float32, device=args.device).reshape(1,7)
            q = robot.data.default_joint_pos.clone()
            q[:,indices] = torch.tensor(row[[ix[f'q_{i}'] for i in range(29)]], dtype=torch.float32, device=args.device)
            robot.write_root_pose_to_sim(root)
            robot.write_root_velocity_to_sim(torch.zeros((1,6), device=args.device))
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            sim.forward()
            pos = root[0,:3].cpu().numpy()
            for view_name, eye, target in camera_views(layout, name, pos, args.doorway_views):
                sim.set_camera_view(eye, target)
                for _ in range(20):
                    sim.render()
                if sim.current_time_step_index != step_index:
                    raise RuntimeError('Unexpected physics progression during static reconstruction')
                pixels = np.asarray(rgb.get_data())
                if pixels.ndim!=3 or pixels.shape[2]<3:
                    raise RuntimeError('No RGB image')
                image = Image.fromarray(pixels[:,:,:3].astype(np.uint8))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0,0,1440,38), fill=(245,244,235))
                draw.text((15,12), f'Measured pose reconstruction | {view_name} | reference {frame/50:.2f} s | no new physics rollout', fill=(30,40,40))
                image.save(args.output/f'{view_name}.png')
                images.append({'file':f'{view_name}.png','trace_row':row_index,'reference_frame':frame,
                               'camera_eye':eye.tolist(),'camera_target':target.tolist(),
                               'sim_time_s':float(row[ix['sim_time']]),'wall_monotonic':float(row[ix['wall_monotonic']])})
        rgb.detach(product); product.destroy()
        if any(digest(path)!=value for path,value in hashes.items()):
            raise RuntimeError('Source changed during rendering')
        report = {'kind':'Static measured-pose reconstruction, not a live dynamics test',
                  'source_sha256':hashes, 'images':images, 'no_dds':True, 'physics_steps_advanced':0}
        (args.output/'provenance.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        # The installed Kit hangs in normal Replicator teardown after product
        # destruction. Match the existing runner's documented standalone exit
        # after all images/provenance have been written; no DDS exists here.
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__=='__main__':
    main()
