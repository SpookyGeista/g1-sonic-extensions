#!/usr/bin/env python3
"""Generate isolated, unarmed G1 film-choreography candidates with Kimodo.

This only generates kinematic references. It never opens DDS, starts a robot
controller, changes a policy, or claims physical tracking has passed. Raw
outputs are immutable; reruns require a fresh output directory. The native
Kimodo G1 exporter performs its own coordinate conversion exactly once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
TOOLCHAIN = REPO / "data/kimodo_toolchain_2026-09-16"
STYLE = (
    "with relaxed shoulders, elbows bent close to the torso and empty hands "
    "held comfortably in front of the lower chest, making small controlled "
    "steps with a steady upper body"
)
CLIPS = {
    "walk_short": {"seconds": 6.0, "distance": 1.6, "turn": 0.0,
                   "prompt": f"A person starts standing, walks forward {STYLE}, then stops in a balanced standing pose."},
    "walk_long": {"seconds": 9.0, "distance": 3.0, "turn": 0.0,
                  "prompt": f"A person starts standing, walks forward slowly {STYLE}, then stops in a balanced standing pose."},
    "turn_left": {"seconds": 7.0, "radius": 1.1, "turn": math.pi/2,
                  "prompt": f"A person starts standing, walks around a gentle left quarter-circle {STYLE}, then stops standing facing left."},
    "turn_right": {"seconds": 7.0, "radius": 1.1, "turn": -math.pi/2,
                   "prompt": f"A person starts standing, walks around a gentle right quarter-circle {STYLE}, then stops standing facing right."},
    "crouch_walk": {"seconds": 12.0, "distance": 3.0, "turn": 0.0,
                    "prompt": "A person starts standing, bends both knees into a comfortable crouch, walks forward slowly while keeping the knees bent and the elbows tucked close with empty hands in front of the chest, then straightens both legs and stops standing."},
    "attentive_idle": {"seconds": 6.0, "distance": 0.0, "turn": 0.0,
                       "prompt": "A person stands balanced with feet planted, relaxed shoulders, elbows bent close to the torso and empty hands near the lower chest, gently looks left and right with small upper-body movements, then returns to the original standing pose."},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def constraints_for(clip, fps):
    """Nominal root waypoints in the documented internal Y-up/+Z-forward frame.

    Initial/final 0.7 seconds request a stationary root, not a planted-foot
    constraint or a balance guarantee. No generated root path is overwritten.
    """
    duration = clip["seconds"]
    total = round(duration*fps)
    frames = sorted({0, total-1, *range(0, total, max(1, round(fps/2)))})
    points, headings = [], []
    for frame in frames:
        progress = max(0.0, min(1.0, (frame/fps-0.7)/(duration-1.4)))
        # Quintic timing eases acceleration at both ends of the requested path.
        progress = progress**3*(10-15*progress+6*progress**2)
        angle = clip["turn"]*progress
        if "radius" in clip:
            radius = clip["radius"]
            sign = 1.0 if clip["turn"] > 0 else -1.0
            points.append([sign*radius*(1-math.cos(angle)), radius*math.sin(abs(angle))])
        else:
            points.append([0.0, clip["distance"]*progress])
        headings.append([math.cos(angle), math.sin(angle)])
    return [{"type": "root2d", "frame_indices": frames,
             "smooth_root_2d": points, "global_root_heading": headings}]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", choices=CLIPS, default=list(CLIPS))
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9162026)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--model", default="Kimodo-G1-RP-v1")
    parser.add_argument("--plan-only", action="store_true", help="Print prompts, no imports/downloads/inference")
    args = parser.parse_args()
    if not 1 <= args.samples <= 4 or not 20 <= args.steps <= 250:
        parser.error("Use 1..4 samples and 20..250 diffusion steps")
    output_dir = args.output.resolve()
    if not output_dir.is_relative_to(REPO / "data"):
        parser.error("Keep generated candidates below this repository's data directory")
    if output_dir.exists():
        parser.error("Output exists; use a new directory to preserve raw generations")
    plan = {"model": args.model, "clips": {k: CLIPS[k] for k in args.clips},
            "samples": args.samples, "seed": args.seed, "steps": args.steps,
            "scope": "Unarmed film choreography; kinematic candidates, not tactical planning or navigation",
            "tracking_validated": False}
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0

    import numpy as np
    import torch
    from kimodo import load_model
    from kimodo.constraints import load_constraints_lst
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.tools import seed_everything

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no hidden, extremely slow CPU generation fallback")
    source = TOOLCHAIN / "kimodo"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    print("KIMODO_LOADING_MODEL", args.model, flush=True)
    model, resolved = load_model(args.model, device="cuda:0", default_family="Kimodo",
                                 return_resolved_name=True)
    if model.skeleton.nbjoints != 34 or "g1" not in resolved.lower():
        raise ValueError("This generator requires the native 34-node G1 skeleton/29-DOF export")
    fps = float(model.fps)
    exporter = MujocoQposConverter(model.skeleton)
    native_xml = Path(exporter.xml_path)
    joint_names = [j.attrib["name"] for j in ET.parse(native_xml).getroot().find("worldbody").findall(".//joint")]
    if len(joint_names) != 29 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid model FPS or native G1 joint schema")
    output_dir.mkdir(parents=True, exist_ok=False)
    plan.update(fps=fps, resolved_model=resolved, source_revision=revision)
    write_json(output_dir / "generation_plan.json", plan)
    completed = []
    for ordinal, name in enumerate(args.clips):
        clip = CLIPS[name]
        folder = output_dir / name
        folder.mkdir()
        constraint_path = folder / "constraints.json"
        write_json(constraint_path, constraints_for(clip, fps))
        constraints = load_constraints_lst(str(constraint_path), model.skeleton)
        seed = args.seed + ordinal
        seed_everything(seed)
        print("KIMODO_GENERATING", name, "fps", fps, "seed", seed, flush=True)
        started = time.monotonic()
        generated = model(clip["prompt"], round(clip["seconds"]*fps),
                          num_samples=args.samples, constraint_lst=constraints,
                          num_denoising_steps=args.steps, first_heading_angle=0.0,
                          post_processing=False, return_numpy=True)
        qpos = exporter.dict_to_qpos(generated, "cpu", mujoco_rest_zero=False)
        if qpos.shape != (args.samples, round(clip["seconds"]*fps), 36) or not np.isfinite(qpos).all():
            raise ValueError("Nonfinite or incorrectly shaped native G1 output")
        for sample in range(args.samples):
            stem = folder / f"{name}_{sample:02d}"
            single = {k: (v[sample] if hasattr(v, "shape") and len(v.shape) > 0
                           and v.shape[0] == args.samples else v) for k, v in generated.items()}
            save_kimodo_npz(str(stem)+".npz", single)
            exporter.save_csv(qpos[sample], str(stem)+".csv")
            meta = {"schema_version": 1, "fps": fps, "fps_basis": "Loaded model.fps",
                    "model_name": args.model, "resolved_model": resolved,
                    "model_source_revision": revision, "seed": seed, "sample_index": sample,
                    "prompt": clip["prompt"], "duration_seconds": clip["seconds"],
                    "frames": len(qpos[sample]), "diffusion_steps": args.steps,
                    "constraints_file": str(constraint_path.relative_to(REPO)),
                    "constraints_sha256": sha256(constraint_path),
                    "joint_names": joint_names, "mujoco_rest_zero": False,
                    "root_quaternion_order": "wxyz", "translation_units": "meters",
                    "joint_angle_units": "radians", "coordinate_system": "z_up_x_forward",
                    "native_model_xml": str(native_xml.relative_to(REPO)),
                    "native_model_xml_sha256": sha256(native_xml),
                    "native_npz_sha256": sha256(str(stem)+".npz"),
                    "native_csv_sha256": sha256(str(stem)+".csv"),
                    "upstream_behavior": "Official native exporter extracts one scalar angle per robot hinge; no post-generation joint-limit clamping. Foot-skate postprocessing disabled as in the official G1 CLI. Downstream limits and physical tracking must be checked separately",
                    "post_generation_pose_changes": "None; official native exporter only",
                    "physical_tracking_validated": False}
            write_json(str(stem)+".meta.json", meta)
            completed.append(str(stem.relative_to(REPO)))
        print("KIMODO_GENERATED", name, "wall_seconds", round(time.monotonic()-started, 2), flush=True)
    write_json(output_dir / "completed.json", {"outputs": completed, "physical_tracking_validated": False})
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
