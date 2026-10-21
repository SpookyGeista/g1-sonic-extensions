"""CPU-only planning/API-contract tests, never model loading or diffusion.

Run with the isolated Kimodo interpreter for the upstream contract tests:
  data/kimodo_toolchain_2026-09-16/.venv/bin/python -B -m unittest discover \
    -s tools/tests -p test_generate_kimodo_demo_clips.py -v
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("kimodo_clip_generator", ROOT / "tools/generate_kimodo_demo_clips.py")
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


class PlanningTests(unittest.TestCase):
    def test_generator_explicitly_selects_kimodo_family(self):
        tree = ast.parse((ROOT / "tools/generate_kimodo_demo_clips.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "load_model"]
        self.assertEqual(len(calls), 1)
        keywords = {kw.arg: kw.value for kw in calls[0].keywords}
        self.assertIn("default_family", keywords)
        self.assertEqual(ast.literal_eval(keywords["default_family"]), "Kimodo")

    def test_official_registry_resolves_display_model_without_loading_model(self):
        registry_path = ROOT / "data/kimodo_toolchain_2026-09-16/kimodo/kimodo/model/registry.py"
        if not registry_path.is_file():
            self.skipTest("Official Kimodo source checkout not present")
        # Registry imports stdlib only. Load by path to avoid importing Kimodo's
        # package __init__, Hugging Face client, Torch, or any text encoder.
        registry_name = "_kimodo_registry_cpu_contract"
        spec = importlib.util.spec_from_file_location(registry_name, registry_path)
        registry = importlib.util.module_from_spec(spec)
        sys.modules[registry_name] = registry  # required by dataclass annotation lookup
        try:
            spec.loader.exec_module(registry)
            self.assertEqual(registry.resolve_model_name("Kimodo-G1-RP-v1", default_family="Kimodo"),
                             "kimodo-g1-rp")
            self.assertEqual(registry.resolve_model_name("Kimodo-G1-SEED-v1", default_family="Kimodo"),
                             "kimodo-g1-seed")
            self.assertEqual(registry.resolve_model_name("G1", default_family="Kimodo"), "kimodo-g1-rp")
        finally:
            del sys.modules[registry_name]

    def test_constraints_frames_and_headings(self):
        for fps in (30.0, 50.0, 60.0):
            for name, clip in GEN.CLIPS.items():
                with self.subTest(fps=fps, clip=name):
                    constraint, = GEN.constraints_for(clip, fps)
                    frames = constraint["frame_indices"]
                    self.assertEqual(frames, sorted(set(frames)))
                    self.assertEqual(frames[0], 0)
                    self.assertEqual(frames[-1], round(clip["seconds"] * fps) - 1)
                    self.assertEqual(len(frames), len(constraint["smooth_root_2d"]))
                    self.assertEqual(len(frames), len(constraint["global_root_heading"]))
                    self.assertEqual(constraint["smooth_root_2d"][0], [0.0, 0.0])
                    self.assertEqual(constraint["global_root_heading"][0], [1.0, 0.0])
                    for heading in constraint["global_root_heading"]:
                        self.assertAlmostEqual(sum(v*v for v in heading), 1.0)

    def test_straight_distance_and_stationary_endpoints(self):
        for name in ("walk_short", "walk_long", "crouch_walk", "attentive_idle"):
            clip = GEN.CLIPS[name]
            constraint, = GEN.constraints_for(clip, 30)
            points = constraint["smooth_root_2d"]
            self.assertEqual(points[-1], [0.0, clip["distance"]])
            self.assertEqual(points[:2], [[0.0, 0.0], [0.0, 0.0]])
            self.assertEqual(points[-2], points[-1])
            self.assertTrue(all(a[1] <= b[1] for a, b in zip(points, points[1:])))

    def test_left_and_right_are_mirrors_with_tangent_headings(self):
        left, = GEN.constraints_for(GEN.CLIPS["turn_left"], 30)
        right, = GEN.constraints_for(GEN.CLIPS["turn_right"], 30)
        radius = GEN.CLIPS["turn_left"]["radius"]
        for (x, z), (rx, rz), (cosine, sine) in zip(
            left["smooth_root_2d"], right["smooth_root_2d"], left["global_root_heading"]
        ):
            self.assertAlmostEqual(x, -rx)
            self.assertAlmostEqual(z, rz)
            self.assertAlmostEqual((x-radius)**2 + z*z, radius*radius)
            # Native MuJoCo transform is (Kz, Kx, Ky): positive Kx is left.
            self.assertAlmostEqual(x, radius * (1-cosine))
            self.assertAlmostEqual(z, radius * sine)
        self.assertAlmostEqual(left["global_root_heading"][-1][1], 1.0)
        self.assertAlmostEqual(right["global_root_heading"][-1][1], -1.0)

    def test_json_writer_never_overwrites(self):
        with tempfile.TemporaryDirectory(prefix="kimodo-generator-test-") as folder:
            target = Path(folder) / "metadata.json"
            GEN.write_json(target, {"original": True})
            with self.assertRaises(FileExistsError):
                GEN.write_json(target, {"original": False})
            self.assertEqual(json.loads(target.read_text()), {"original": True})

    def test_plan_only_does_not_create_output(self):
        target = ROOT / "data" / ("kimodo-plan-test-" + uuid.uuid4().hex)
        result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/generate_kimodo_demo_clips.py"),
                                 "--output", str(target), "--clips", "walk_short", "turn_left",
                                 "--plan-only"], check=True, capture_output=True, text=True)
        self.assertFalse(target.exists())
        plan = json.loads(result.stdout)
        self.assertEqual(list(plan["clips"]), ["walk_short", "turn_left"])
        self.assertFalse(plan["tracking_validated"])

    def test_existing_output_is_rejected_before_model_import(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/generate_kimodo_demo_clips.py"),
                                 "--output", str(ROOT / "data"), "--plan-only"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Output exists", result.stderr)


@unittest.skipUnless(importlib.util.find_spec("kimodo"), "Use isolated Kimodo environment for API contract checks")
class UpstreamCPUContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from kimodo.constraints import load_constraints_lst
        from kimodo.exports.mujoco import MujocoQposConverter
        from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
        from kimodo.skeleton.registry import build_skeleton
        cls.torch = torch
        cls.skeleton = build_skeleton(34).to("cpu")
        cls.rep = KimodoMotionRep(cls.skeleton, 30.0)
        cls.converter = MujocoQposConverter(cls.skeleton)
        cls.constraints = load_constraints_lst(GEN.constraints_for(GEN.CLIPS["turn_left"], 30), cls.skeleton)

    def test_shared_constraints_repeated_across_batch(self):
        lengths = self.torch.tensor([210, 210])
        observed, mask = self.rep.create_conditions_from_constraints_batched(
            self.constraints, lengths, to_normalize=False, device="cpu")
        self.assertEqual(tuple(observed.shape), (2, 210, self.rep.motion_rep_dim))
        self.assertTrue(self.torch.equal(observed[0], observed[1]))
        self.assertTrue(self.torch.equal(mask[0], mask[1]))
        self.assertTrue(mask.any())

    def test_model_call_keeps_explicit_single_sample_batch(self):
        from kimodo.model.kimodo_model import Kimodo
        torch = self.torch
        for samples in (1, 2):
            captures = {}

            def conditions(constraints, lengths, *, to_normalize, device):
                captures["normalize_requested"] = to_normalize
                return self.rep.create_conditions_from_constraints_batched(
                    constraints, lengths, to_normalize=False, device=device)

            def generate(texts, frames, **kwargs):
                captures.update(kwargs)
                return torch.zeros((len(texts), int(frames), 1))

            fake = SimpleNamespace(device="cpu", skeleton=self.skeleton, _generate=generate,
                                   motion_rep=SimpleNamespace(
                                       create_conditions_from_constraints_batched=conditions,
                                       inverse=lambda motion, **_: {"root_positions": torch.zeros((*motion.shape[:2], 3))}))
            # Exercises official public __call__ plumbing only; no denoiser/text encoder is constructed.
            output = Kimodo.__call__(fake, "A person walks.", 210, num_denoising_steps=100,
                                     num_samples=samples, constraint_lst=self.constraints,
                                     first_heading_angle=0.0, post_processing=False, return_numpy=True)
            self.assertEqual(output["root_positions"].shape, (samples, 210, 3))
            self.assertTrue(captures["normalize_requested"])
            self.assertEqual(tuple(captures["first_heading_angle"].shape), (samples,))
            self.assertEqual(tuple(captures["observed_motion"].shape[:2]), (samples, 210))

    def test_native_export_uses_z_up_x_forward_and_wxyz(self):
        torch = self.torch
        local = torch.eye(3).repeat(2, 3, 34, 1, 1)
        root = torch.tensor([[[1.0, 2.0, 3.0]]]).repeat(2, 3, 1)
        qpos = self.converter.dict_to_qpos({"local_rot_mats": local, "root_positions": root},
                                         "cpu", mujoco_rest_zero=False)
        self.assertEqual(qpos.shape, (2, 3, 36))
        self.assertEqual(qpos[0, 0, :3].tolist(), [3.0, 1.0, 2.0])
        self.assertEqual(qpos[0, 0, 3:7].tolist(), [1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
