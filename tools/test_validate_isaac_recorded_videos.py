import unittest

import numpy as np

from validate_isaac_recorded_videos import (
    audit_edit, check_probe, compare_rgb, comparison_indices, rgb_hash_audit,
)


class VideoValidatorTests(unittest.TestCase):
    def setUp(self):
        self.edit = [
            {'start_s': 0, 'end_s': .08, 'camera': 'top'},
            {'start_s': .08, 'end_s': .16, 'camera': 'first'},
            {'start_s': .16, 'end_s': .24, 'camera': 'chase'},
        ]

    def test_full_edit_requires_three_actual_views(self):
        self.assertTrue(audit_edit(self.edit, 6, 25, True)['passed'])
        self.assertFalse(audit_edit(self.edit, 2, 25, True)['passed'])
        self.assertTrue(audit_edit(self.edit, 2, 25, False)['passed'])

    def test_edit_cut_neighborhoods_and_gap(self):
        self.assertEqual(comparison_indices(self.edit, 6, 25), list(range(6)))
        self.edit[1]['start_s'] = .09
        self.assertFalse(audit_edit(self.edit, 6, 25, True)['passed'])

    def test_stale_rgb_frames_detected(self):
        hashes = [{name: digit*64 for name in ('top', 'first', 'chase')} for digit in ('1', '2')]
        self.assertTrue(rgb_hash_audit(hashes, 2)['passed'])
        hashes[1]['top'] = hashes[0]['top']
        self.assertFalse(rgb_hash_audit(hashes, 2)['passed'])

    def test_codec_tolerance_rejects_wrong_view(self):
        source = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertTrue(compare_rgb(source+2, source)['passed'])
        self.assertFalse(compare_rgb(source+40, source)['passed'])

    def test_stream_properties_and_audio(self):
        probe = {'streams': [{'codec_type': 'video', 'width': 1280, 'height': 720,
            'avg_frame_rate': '25/1', 'r_frame_rate': '25/1', 'nb_read_frames': '12',
            'nb_frames': '12', 'duration': '.480000', 'codec_name': 'h264', 'pix_fmt': 'yuv420p'}],
            'format': {'duration': '.480000'}}
        self.assertTrue(check_probe(probe, 12, 25, [1280, 720])['passed'])
        probe['streams'].append({'codec_type': 'audio'})
        self.assertFalse(check_probe(probe, 12, 25, [1280, 720])['passed'])


if __name__ == '__main__':
    unittest.main()
