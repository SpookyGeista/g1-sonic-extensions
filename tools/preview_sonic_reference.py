#!/usr/bin/env python3
"""Offline MuJoCo kinematic preview; NEVER connects to DDS or sends robot commands.

Shows recorded reference poses, not SONIC's physical policy tracking.
"""
import argparse
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from convert_bones_to_sonic_reference import MJ_TO_IL, BODY_INDEXES_MJ
from merge_sonic_reference_motions import load_motion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--video", type=Path)
    p.add_argument("--contact-sheet", type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args()
    if not args.interactive:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    motion = load_motion(args.motion)
    xml = Path(__file__).resolve().parents[1]/"gear_sonic_deploy/g1/g1_29dof.xml"
    tree = ET.parse(xml).getroot()
    tree.find("compiler").set("meshdir",str(xml.parent/"meshes"))
    world = tree.find("worldbody")
    floors=tree.findall(".//geom[@type='plane']")
    if not floors:
        floors=[ET.SubElement(world,"geom",name="preview_floor",type="plane",size="10 10 .1",rgba=".18 .21 .24 1")]
    for floor in floors: floor.set("group","2")
    ET.SubElement(world,"light",pos="2 -2 4",dir="-.3 .3 -1",directional="true")
    visual = ET.SubElement(tree,"visual")
    ET.SubElement(visual,"global",offwidth="960",offheight="540")
    model=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding="unicode"))
    data=mujoco.MjData(model)
    option=mujoco.MjvOption()
    option.geomgroup[0]=0  # Collision meshes duplicate the visual meshes.
    frames=len(motion["joint_pos"])
    def pose(i):
        data.qpos[:3]=motion["body_pos"][i,0]
        data.qpos[3:7]=motion["body_quat"][i,0]
        data.qpos[7:]=motion["joint_pos"][i,MJ_TO_IL]
        mujoco.mj_forward(model,data)
    if args.interactive:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model,data) as viewer:
            viewer.opt.geomgroup[0]=0
            start=time.monotonic()
            while viewer.is_running():
                frame=min(int((time.monotonic()-start)*50),frames-1)
                pose(frame)
                viewer.cam.lookat[:]=motion["body_pos"][frame,0]
                viewer.cam.distance=2.6
                viewer.sync()
                time.sleep(.01)
        return
    import cv2
    camera=mujoco.MjvCamera()
    camera.azimuth=140
    camera.elevation=-18
    camera.distance=2.7
    width,height=480,360
    renderer=mujoco.Renderer(model,height=height,width=width)
    writer=None
    if args.video:
        args.video.parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(args.video),cv2.VideoWriter_fourcc(*"mp4v"),25,(width,height))
        if not writer.isOpened():raise RuntimeError("Could not open MP4 writer")
    selected=set(np.linspace(0,frames-1,8,dtype=int).tolist())
    pictures=[]
    ankle_ids=[model.body(n).id for n in ("left_ankle_roll_link","right_ankle_roll_link")]
    foot_geoms=np.array([i for i in range(model.ngeom) if model.geom_bodyid[i] in ankle_ids and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE])
    minimum_foot=[]
    fk_error=0.
    try:
        for i in range(frames):
            pose(i)
            minimum_foot.append(float((data.geom_xpos[foot_geoms,2]-model.geom_size[foot_geoms,0]).min()))
            fk_error=max(fk_error,float(abs(data.xpos[BODY_INDEXES_MJ+1]-motion["body_pos"][i]).max()))
            if (writer and i%2==0) or i in selected:
                camera.lookat[:]=motion["body_pos"][i,0]
                camera.lookat[2]=.65
                renderer.update_scene(data,camera=camera,scene_option=option)
                frame=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
                cv2.putText(frame,f"t = {i/50:.2f} s    reference only",(15,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1,cv2.LINE_AA)
                if writer and i%2==0:writer.write(frame)
                if i in selected:pictures.append(frame.copy())
    finally:
        if writer:writer.release()
        renderer.close()
    if args.contact_sheet:
        args.contact_sheet.parent.mkdir(parents=True,exist_ok=True)
        sheet=np.concatenate([np.concatenate(pictures[:4],axis=1),np.concatenate(pictures[4:],axis=1)],axis=0)
        if not cv2.imwrite(str(args.contact_sheet),sheet):raise RuntimeError("Could not save sheet")
    assert fk_error<1e-4,f"MuJoCo FK inconsistent with reference: {fk_error}"
    report={"motion":str(args.motion),"frames":frames,"mujoco_fk_max_position_error_m":fk_error,
            "minimum_foot_collision_sphere_z_m":min(minimum_foot),
            "maximum_minimum_foot_sphere_z_m":max(minimum_foot),
            "worst_foot_penetration_frame":int(np.argmin(minimum_foot)),
            "is_policy_tracking_test":False,
            "note":"Reference poses only. Negative foot height indicates source/edited reference penetration, not a physics simulation result."}
    if args.report:args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
