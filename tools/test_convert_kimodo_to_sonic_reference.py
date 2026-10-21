#!/usr/bin/env python3
"""CPU-only synthetic tests; these are NOT generated Kimodo clips or stability tests.

Run: .venv_sim/bin/python -m unittest discover -s tools \
       -p 'test_convert_kimodo_to_sonic_reference.py' -v
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

from convert_bones_to_sonic_reference import BODY_INDEXES_MJ, G1Kinematics, IL_TO_MJ
from convert_kimodo_to_sonic_reference import (
    DEFAULT_KIMODO, DEFAULT_MJCF, EXPECTED_CONVENTIONS,
    audit_skeletons, convert_one, resample_qpos, resolve_fps,
)


def synthetic_qpos(frames=31):
    """Small articulated movement, translating and yawing; every hinge in limits."""
    t = np.linspace(0, 1, frames)
    qpos = np.zeros((frames, 36))
    qpos[:, :3] = np.column_stack((0.15 * t, 0.03 * t, 0.79 + 0.005 * np.sin(2 * np.pi * t)))
    yaw = 0.2 * t
    qpos[:, 3] = np.cos(yaw / 2)
    qpos[:, 6] = np.sin(yaw / 2)
    base = np.zeros(29)
    base[[0, 6]] = -0.15
    base[[3, 9]] = 0.30
    base[[4, 10]] = -0.15
    base[[15, 22]] = 0.15
    base[16], base[23] = 0.15, -0.15
    base[[18, 25]] = 0.5
    qpos[:, 7:] = base + 0.01 * np.sin(t[:, None] * 2 * np.pi + np.arange(29)[None] * 0.1)
    return qpos


class KimodoConverterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kimodo_adapter_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / "synthetic_not_generated.csv"
        self.qpos = synthetic_qpos()
        np.savetxt(self.source, self.qpos, delimiter=",")
        self.fk = G1Kinematics(DEFAULT_MJCF)
        self.meta = {"fps": 30, "synthetic_test_data": True,
                     "joint_names": self.fk.joint_names, **EXPECTED_CONVENTIONS}
        self.write_metadata()
        self.output = self.folder / "reference"

    def write_metadata(self):
        self.source.with_suffix(".meta.json").write_text(json.dumps(self.meta))

    def test_six_csv_shapes_mapping_durations_and_source_preserved(self):
        original = self.source.read_bytes()
        report = convert_one(self.source, self.output)
        self.assertEqual(report["output_frames"], 51)
        self.assertEqual(report["output_duration_s"], 1.0)
        self.assertEqual(report["source_duration_s"], 1.0)
        self.assertTrue(report["joint_limits_ok"])
        self.assertFalse(report["dynamic_simulation_tested"])
        widths = {"joint_pos": 29, "joint_vel": 29, "body_pos": 42,
                  "body_quat": 56, "body_lin_vel": 42, "body_ang_vel": 42}
        arrays = {}
        for name, width in widths.items():
            arrays[name] = np.loadtxt(self.output / f"{name}.csv", delimiter=",", skiprows=1)
            self.assertEqual(arrays[name].shape, (51, width))
            self.assertTrue(np.isfinite(arrays[name]).all())
        np.testing.assert_allclose(arrays["joint_pos"][[0, -1]], self.qpos[[0, -1], 7:][:, IL_TO_MJ], atol=5.01e-7)
        np.testing.assert_allclose(arrays["body_pos"][[0, -1], :3], self.qpos[[0, -1], :3], atol=5.01e-7)
        np.testing.assert_allclose(np.linalg.norm(arrays["body_quat"].reshape(51, 14, 4), axis=-1), 1, atol=1e-6)
        self.assertEqual(original, self.source.read_bytes())
        provenance = json.loads((self.output / "conversion_metadata.json").read_text())
        self.assertEqual(provenance["pose_clipping_or_smoothing"], "none")
        self.assertEqual(provenance["source_fps_evidence"], ["sidecar:fps"])
        self.assertEqual(provenance["validation"]["trajectory"]["end_root_xyz_m"], self.qpos[-1, :3].tolist())

    def test_independent_real_mujoco_fk_and_matching_native_angular_conventions(self):
        target = mujoco.MjModel.from_xml_path(str(DEFAULT_MJCF))
        native_xml = DEFAULT_KIMODO / "kimodo/assets/skeletons/g1skel34/xml/g1.xml"
        native = mujoco.MjModel.from_xml_path(str(native_xml))
        td, nd = mujoco.MjData(target), mujoco.MjData(native)
        out, _ = resample_qpos(self.qpos, 30)
        positions, rotations = self.fk.forward(out[:, :3], out[:, 3:7], out[:, 7:])
        for frame in (0, 17, 50):
            td.qpos[:] = nd.qpos[:] = out[frame]
            mujoco.mj_forward(target, td)
            mujoco.mj_forward(native, nd)
            # Compare all 30 articulated-body world rotations, not just the 14 exported ones.
            for body_name in self.fk.names:
                ti = target.body(body_name).id
                ni = native.body(body_name).id
                self.assertAlmostEqual(abs(float(np.dot(td.xquat[ti], nd.xquat[ni]))), 1.0, places=10)
            for selection, joint_body in enumerate(BODY_INDEXES_MJ):
                body_id = target.body(self.fk.names[joint_body]).id
                np.testing.assert_allclose(positions[frame, selection], td.xpos[body_id], atol=1e-9)
                self.assertAlmostEqual(abs(float(np.dot(rotations[frame, selection], td.xquat[body_id]))), 1.0, places=10)
        audit = audit_skeletons(DEFAULT_KIMODO, DEFAULT_MJCF, self.fk)
        self.assertTrue(audit["joint_names_order_axes_and_rest_rotations_match"])
        self.assertEqual(len(audit["morphology_offset_differences"]), 4)

    def test_missing_and_conflicting_fps_fail(self):
        self.meta.pop("fps")
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "Unknown source FPS"):
            convert_one(self.source, self.output)
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, "Conflicting source FPS"):
            resolve_fps({"fps": 30}, 120)
        self.assertEqual(resolve_fps({}, 30)[0], 30)

    def test_limit_policy_rejects_without_clipping_or_reports_explicitly(self):
        self.qpos[10, 7] = 4.0
        np.savetxt(self.source, self.qpos, delimiter=",")
        with self.assertRaisesRegex(ValueError, "Source joint limit violation"):
            convert_one(self.source, self.output)
        self.assertFalse(self.output.exists())
        report = convert_one(self.source, self.output, joint_limit_policy="report")
        self.assertFalse(report["joint_limits_ok"])
        # 30 Hz frame10 coincides only approximately with 50 Hz; interpolation preserves
        # the out-of-range movement instead of quietly clipping it to the joint limit.
        values = np.loadtxt(self.output / "joint_pos.csv", delimiter=",", skiprows=1)
        self.assertGreater(values[:, 0].max(), 3.0)

    def test_bad_quaternion_nonfinite_and_schema_fail(self):
        for bad in ("zero_quaternion", "nan", "wrong_width"):
            with self.subTest(bad=bad):
                q = self.qpos.copy()
                if bad == "zero_quaternion":
                    q[0, 3:7] = 0
                elif bad == "nan":
                    q[1, 7] = np.nan
                else:
                    q = q[:, :-1]
                np.savetxt(self.source, q, delimiter=",")
                with self.assertRaises(ValueError):
                    convert_one(self.source, self.output)
                self.assertFalse(self.output.exists())

    def test_unrecognized_units_rest_convention_and_joint_order_fail(self):
        for key, value in (("translation_units", "centimeters"), ("mujoco_rest_zero", True),
                           ("root_quaternion_order", "xyzw"), ("joint_names", self.fk.joint_names[::-1])):
            with self.subTest(key=key):
                old = self.meta[key]
                self.meta[key] = value
                self.write_metadata()
                with self.assertRaises(ValueError):
                    convert_one(self.source, self.output)
                self.meta[key] = old

    def test_source_xml_axis_or_rest_change_fails(self):
        copy_root = self.folder / "kimodo_copy"
        relative_xml = Path("kimodo/assets/skeletons/g1skel34/xml/g1.xml")
        xml_path = copy_root / relative_xml
        xml_path.parent.mkdir(parents=True)
        original_xml = ET.parse(DEFAULT_KIMODO / relative_xml)
        joint = original_xml.getroot().find("worldbody").find(".//joint")
        joint.set("axis", "0 0 1")
        original_xml.write(xml_path)
        with self.assertRaisesRegex(ValueError, "Hinge-axis mismatch"):
            audit_skeletons(copy_root, DEFAULT_MJCF, self.fk)

    def test_stale_sidecar_hashes_fail_before_output(self):
        for key in ("native_csv_sha256", "native_model_xml_sha256"):
            with self.subTest(key=key):
                self.meta[key] = "0" * 64
                self.write_metadata()
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    convert_one(self.source, self.output)
                self.assertFalse(self.output.exists())
                self.meta.pop(key)

    def test_antipodal_quaternions_no_flip_no_nan(self):
        opposite = self.qpos.copy()
        opposite[1::2, 3:7] *= -1
        result, times = resample_qpos(opposite, 30)
        reference, expected_times = resample_qpos(self.qpos, 30)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(abs(np.sum(result[:, 3:7] * reference[:, 3:7], axis=1)), 1, atol=1e-12)
        np.testing.assert_array_equal(times, expected_times)

    def test_no_overwrite_existing_reference(self):
        convert_one(self.source, self.output)
        with self.assertRaises(FileExistsError):
            convert_one(self.source, self.output)


if __name__ == "__main__":
    unittest.main()
