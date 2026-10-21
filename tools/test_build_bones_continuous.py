import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_bones_continuous import collapse_holds, identical_edges, longest_true_run, build
from build_bones_crouch_pair import crossfade


def motion(values):
    x=np.asarray(values,dtype=float)
    return {'root':np.c_[x,np.zeros_like(x),np.full_like(x,.77)],
            'quat':np.tile([1.,0.,0.,0.],(len(x),1)),
            'joints':np.tile(x[:,None]*.1,(1,29))}


class ContinuousTest(unittest.TestCase):
    def test_exact_holds_collapsed_without_changing_moving_samples(self):
        source=motion([0,0,0,1,2,2,2,3,4,4,4])
        before={k:v.copy() for k,v in source.items()}
        out,indices,runs=collapse_holds(source)
        np.testing.assert_array_equal(indices,[0,3,4,7,8])
        np.testing.assert_array_equal(out['root'][:,0],[0,1,2,3,4])
        self.assertEqual(sum(r['removed_frames'] for r in runs),6)
        for key in source:np.testing.assert_array_equal(source[key],before[key])

    def test_slow_motion_is_not_a_duplicate(self):
        source=motion([0,1e-7,2e-7,3e-7])
        out,indices,runs=collapse_holds(source)
        self.assertEqual(len(out['root']),4)
        self.assertEqual(runs,[])

    def test_antipodal_quaternions_count_as_same_pose(self):
        source=motion([0,0,0,1])
        source['quat'][1]*=-1
        out,indices,runs=collapse_holds(source)
        np.testing.assert_array_equal(indices,[0,3])

    def test_moving_overlap_does_not_insert_frames_or_change_sources(self):
        first,second=motion(np.arange(20)*.01),motion(np.arange(20)*.01)
        second['root'][:,2]=.79
        saved={k:v.copy() for k,v in second.items()}
        out=crossfade(first,second,5)
        self.assertEqual(len(out['root']),35)
        self.assertFalse(identical_edges(out).any())
        self.assertAlmostEqual(out['root'][14,2],.77)
        self.assertAlmostEqual(out['root'][19,2],.79)
        for key in second:np.testing.assert_array_equal(second[key],saved[key])

    def test_longest_run(self):
        self.assertEqual(longest_true_run([1,1,0,1,1,1,0]),3)
        self.assertEqual(longest_true_run([]),0)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'data') as directory:
            with self.assertRaisesRegex(ValueError,'NEW output'):
                build('does-not-exist.json',directory)


if __name__=='__main__':unittest.main()
