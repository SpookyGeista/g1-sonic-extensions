import unittest
import numpy as np
from render_isaac_recorded_videos import camera_rotations,validate_edit,chosen_view,interp

class ExportTests(unittest.TestCase):
    def test_mix_covers_time_without_reordering(self):
        edit=[{'camera':'top','start_s':0,'end_s':3},{'camera':'first','start_s':3,'end_s':5}]
        validate_edit(edit,5)
        self.assertEqual(chosen_view(edit,2.99),'top')
        self.assertEqual(chosen_view(edit,3),'first')
        with self.assertRaises(ValueError):validate_edit(edit,6)
        edit[1]['start_s']=3.1
        with self.assertRaises(ValueError):validate_edit(edit,5)

    def test_eye_convention(self):
        r=[{'first':{'eye':[0,0,1],'target':[1,0,1],'up':[0,0,1]}}]
        matrix=camera_rotations(r,'first').as_matrix()[0]
        np.testing.assert_allclose(-matrix[:,2],[1,0,0],atol=1e-10)
        np.testing.assert_allclose(matrix[:,1],[0,0,1],atol=1e-10)

    def test_interpolate(self):
        np.testing.assert_allclose(interp([0,1],[[0,2],[2,4]],[.25]),[[.5,2.5]])

if __name__=='__main__':unittest.main()
