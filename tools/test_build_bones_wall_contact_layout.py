import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from build_bones_wall_contact_layout import build

SOURCE = Path(__file__).resolve().parents[1] / 'data/isaac_sonic_apartment_2026-09-16/layout.json'
sys.path.insert(0, str(SOURCE.parent))
from layout_geometry import collision_boxes_from_layout


class ContactLayoutTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'layout.json'
        self.original = SOURCE.read_bytes()

    def test_identity_preserves_collision_geometry(self):
        build(SOURCE, self.output, 2.7, 4.25)
        old = collision_boxes_from_layout(json.loads(self.original))
        new = collision_boxes_from_layout(json.loads(self.output.read_text()))
        self.assertEqual([b['prim'] for b in old], [b['prim'] for b in new])
        for a, b in zip(old, new):
            for key in ('aabb_min', 'aabb_max'):
                for x, y in zip(a[key], b[key]):
                    self.assertAlmostEqual(x, y, places=10)

    def test_gate_and_motion_unchanged_and_walls_connected(self):
        build(SOURCE, self.output, 2.2, 3.89, .65, 4.4)
        new = json.loads(self.output.read_text())
        old = json.loads(self.original)
        gate = lambda d: next(p for p in d['portals'] if p['name'] == 'CrawlGate50cm')
        self.assertEqual(gate(old), gate(new))
        self.assertEqual(SOURCE.read_bytes(), self.original)
        self.assertFalse(new['wall_fit_provenance']['motion_modified'])
        self.assertEqual(new['wall_fit_provenance']['source_sha256'], hashlib.sha256(self.original).hexdigest())
        walls = {w['name']: w for w in new['walls']}
        self.assertAlmostEqual(walls['CoverWallWestNorth']['center'][1]-walls['CoverWallWestNorth']['size'][1]/2, .65+.12)
        self.assertAlmostEqual(walls['CoverWallUpperRight']['center'][0]-walls['CoverWallUpperRight']['size'][0]/2, 4.4+.12)

    def test_existing_output_refused(self):
        build(SOURCE, self.output, 2.2, 3.89)
        saved = self.output.read_bytes()
        with self.assertRaises(ValueError):
            build(SOURCE, self.output, 2.3, 3.95)
        self.assertEqual(saved, self.output.read_bytes())

    def test_invalid_planes_and_narrow_openings_refused(self):
        for args in ((9,4), (2.2,1), (2.2,3.9,-.8,4.45), (2.2,3.9,.7,3.5)):
            with self.assertRaises(ValueError):
                build(SOURCE, self.output, *args)
            self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
