import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import convert_bones_to_sonic_reference as conv
from build_bones_crawl_extension import blend_prone_safe, build, checked_range, rotation_matrix, yaw_alignment


def quat(roll=0,pitch=0,yaw=0):
    return conv.euler_extrinsic_xyz_to_wxyz(np.deg2rad([roll,pitch,yaw]))


def motion(pitch=0,yaw=0,height=.77):
    return dict(root=np.c_[np.arange(40)*.01,np.zeros(40),np.full(40,height)],
                quat=np.tile(quat(pitch=pitch,yaw=yaw),(40,1)),
                joints=np.zeros((40,29)))


class CrawlBuildTest(unittest.TestCase):
    def test_exact_prone_yaw_alignment_avoids_euler_singularity(self):
        for pitch in (0,89.999,90,90.001,95):
            source=quat(pitch=pitch,yaw=-37)
            target=quat(pitch=pitch,yaw=123)
            q,angle=yaw_alignment(target,source)
            self.assertAlmostEqual(np.rad2deg(angle),160,places=8)
            self.assertAlmostEqual(abs(np.dot(conv.quat_mul(q,source),target)),1,places=10)

    def test_quaternion_sign_does_not_change_alignment(self):
        a,b=quat(pitch=90,yaw=140),quat(pitch=90,yaw=10)
        qa,_=yaw_alignment(a,b);qb,_=yaw_alignment(-a,-b)
        np.testing.assert_allclose(qa,qb,atol=1e-12)

    def test_degenerate_orientation_rejected(self):
        with self.assertRaises(ValueError):yaw_alignment(quat(),quat(roll=180))

    def test_prone_overlap_no_z_translation_no_source_mutation(self):
        a=motion(pitch=90,yaw=90,height=.11)
        b=motion(pitch=90,yaw=-90,height=.13)
        saved={k:v.copy() for k,v in b.items()}
        result,seam=blend_prone_safe(a,b,15)
        self.assertEqual(len(result['root']),65)
        self.assertEqual(seam['z_translation_m'],0)
        self.assertAlmostEqual(result['root'][25,2],.11)
        self.assertAlmostEqual(result['root'][39,2],.13)
        np.testing.assert_allclose(result['root'][40:,2],.13)
        for key in a:np.testing.assert_array_equal(result[key][:25],a[key][:25])
        for key in b:np.testing.assert_array_equal(b[key],saved[key])
        np.testing.assert_allclose(np.linalg.norm(result['quat'],axis=1),1,atol=1e-12)

    def test_rotation_preserves_local_z_heading_in_prone_pose(self):
        q=quat(pitch=90,yaw=30)
        r=rotation_matrix(q)
        np.testing.assert_allclose(r[:,2],[np.sqrt(3)/2,.5,0],atol=1e-12)
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)

    def test_invalid_ranges_and_overlap_rejected(self):
        a=motion()
        for interval in ([-1,20],[20,50],[20,10],[0,0]):
            with self.assertRaises(ValueError):checked_range(a,interval)
        for count in (0,2,40):
            with self.assertRaises(ValueError):blend_prone_safe(a,a,count)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'data') as directory:
            with self.assertRaisesRegex(ValueError,'NEW output'):
                build('missing.json',directory)


if __name__=='__main__':unittest.main()
