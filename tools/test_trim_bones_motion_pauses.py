import unittest
import numpy as np
from trim_bones_motion_pauses import time_map,retime,remap_timeline

class RetimeTests(unittest.TestCase):
    def test_monotonic_endpoints_outside(self):
        t=time_map(100,[{'source_frames_inclusive':[20,60],'output_intervals':10}])
        self.assertEqual(len(t),70)
        self.assertTrue(np.all(np.diff(t)>0))
        np.testing.assert_array_equal(t[:20],np.arange(20))
        np.testing.assert_array_equal(t[30:],np.arange(60,100))

    def test_reject_overlap_and_endpoints(self):
        for window in ([{'source_frames_inclusive':[0,100],'output_intervals':6}],
                       [{'source_frames_inclusive':[5,30],'output_intervals':6},{'source_frames_inclusive':[25,40],'output_intervals':6}]):
            with self.assertRaises(ValueError):time_map(100,window)

    def test_quaternion_sign_and_no_spatial_change(self):
        motion={'root':np.arange(18,dtype=float).reshape(6,3),
                'joints':np.zeros((6,29)), 'quat':np.tile([1.,0,0,0],(6,1))}
        motion['quat'][3:]*=-1
        t=np.array([0,1.5,2.5,5])
        r=retime(motion,t)
        np.testing.assert_allclose(np.linalg.norm(r['quat'],axis=1),1)
        np.testing.assert_array_equal(r['root'][[0,-1]],motion['root'][[0,-1]])
        np.testing.assert_allclose(r['root'][1],(motion['root'][1]+motion['root'][2])/2)

    def test_timeline_contiguous(self):
        t=time_map(100,[{'source_frames_inclusive':[20,60],'output_intervals':10}])
        stages=[{'start_frame':0,'end_frame_exclusive':40},{'start_frame':40,'end_frame_exclusive':100}]
        out=remap_timeline(stages,t)
        self.assertEqual(out[0]['end_frame_exclusive'],out[1]['start_frame'])
        self.assertEqual(out[1]['end_frame_exclusive'],len(t))

if __name__=='__main__':unittest.main()
