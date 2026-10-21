#!/usr/bin/env python3
"""Fit a separate apartment's connected walls, preserving the original motion.

Door edges are CLEAR opening edges; wall segments meet the outside of their
12-cm jambs. This is geometry construction, not a contact or stability approval.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


def build(source, output, left_x, upper_y, left_door_top=.7, upper_door_right=4.45):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new layout; existing candidates/traces are immutable')
    raw = source.read_bytes()
    layout = copy.deepcopy(json.loads(raw))
    walls = {w['name']: w for w in layout['walls']}
    portals = {p['name']: p for p in layout['portals']}
    rooms = {r['name']: r for r in layout['rooms']}
    left_bottom, upper_left, trim = -.9, 3.05, .12
    if not (1.8 <= left_x <= 2.7 and 3.5 <= upper_y <= 4.25):
        raise ValueError('Wall plane outside bounded fitting region')
    if left_door_top-left_bottom < 1.0 or upper_door_right-upper_left < 1.0:
        raise ValueError('Do not narrow these openings below one metre')

    def vertical(name, x, bottom, top):
        assert top > bottom
        walls[name]['center'][:2] = [x, (bottom+top)/2]
        walls[name]['size'][:2] = [.18, top-bottom]

    def horizontal(name, y, left, right):
        assert right > left
        walls[name]['center'][:2] = [(left+right)/2, y]
        walls[name]['size'][:2] = [right-left, .18]

    portals['CoverDoor01']['center_xy'] = [left_x, (left_bottom+left_door_top)/2]
    portals['CoverDoor01']['clear_width'] = left_door_top-left_bottom
    vertical('CoverWallWestSouth', left_x, -1.4, left_bottom-trim)
    vertical('CoverWallWestNorth', left_x, left_door_top+trim, 7.15)
    horizontal('WestRoomSouthRight', 3.2, 1.17, left_x)
    portals['CoverDoor02']['center_xy'] = [(upper_left+upper_door_right)/2, upper_y]
    portals['CoverDoor02']['clear_width'] = upper_door_right-upper_left
    horizontal('CoverWallUpperLeft', upper_y, left_x, upper_left-trim)
    horizontal('CoverWallUpperRight', upper_y, upper_door_right+trim, 9.75)
    horizontal('UpperWallFarRight', upper_y, 11.19, 12.2)
    portals['Room02NorthDoor']['center_xy'][1] = upper_y
    vertical('EastRoomDividerTop', 8.15, 2.12, upper_y)
    vertical('CrawlGateSideSouth', 8.4, upper_y, 4.61)

    for name in ('Entry', 'WestRoom'):
        rooms[name]['bounds_xy'][1][0] = left_x-.09
    for name in ('MainCorridor', 'UpperPassage'):
        rooms[name]['bounds_xy'][0][0] = left_x+.09
    for name in ('MainCorridor', 'EastRoomA', 'EastRoomB'):
        rooms[name]['bounds_xy'][1][1] = upper_y-.09
    rooms['UpperPassage']['bounds_xy'][0][1] = upper_y+.09

    layout['title'] = 'Wohnungsgrundriss mit koerpernahen Deckungswaenden — separate Kontaktprobe'
    layout['design_status'] = 'Candidate only; requires fresh SONIC/Isaac wall-contact trial'
    layout['wall_fit_provenance'] = {
        'source': str(source), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'left_wall_x': left_x, 'upper_wall_y': upper_y,
        'left_clear_door_top_y': left_door_top, 'upper_clear_door_right_x': upper_door_right,
        'motion_modified': False, 'crawl_gate_modified': False,
        'claim': 'Connected geometry only; no inference of physical contact from AABB overlap.',
    }
    sys.path.insert(0, str(source.parent))
    from layout_geometry import validate_layout, collision_boxes_from_layout
    validate_layout(layout)
    collision_boxes_from_layout(layout)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(layout, indent=2, ensure_ascii=False)+'\n')
    return {'layout': str(output), **layout['wall_fit_provenance']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--left-x', type=float, required=True)
    p.add_argument('--upper-y', type=float, required=True)
    p.add_argument('--left-door-top', type=float, default=.7)
    p.add_argument('--upper-door-right', type=float, default=4.45)
    a = p.parse_args()
    print(json.dumps(build(a.source, a.output, a.left_x, a.upper_y,
                           a.left_door_top, a.upper_door_right), indent=2))
