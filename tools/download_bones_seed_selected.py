#!/usr/bin/env python3
"""Download only the selected BONES-SEED G1 CSV motions.

The Hugging Face repository stores all G1 CSVs inside a single 23.5 GB
``g1.tar.gz`` archive.  The official BONES-SEED viewer exposes the same files
individually, so this script uses its metadata and G1 CSV endpoints to avoid
downloading the complete archive.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import urllib.parse
import urllib.request


VIEWER_API = "https://seed-viewer.bones.studio/api"

SELECTED_MOTIONS = (
    ("Taking cover", "h_b_w_stand_to_crouch_270_start_003__A304"),
    ("Taking cover", "h_b_w_stand_to_crouch_270_start_002__A304"),
    ("Taking cover", "h_b_w_stand_to_crouch_270_start_001__A304"),
    ("Taking cover", "h_b_w_crouch_270_start_003__A550"),
    ("Getting out of cover", "h_b_w_crouch_270_stop_003__A304"),
    ("Getting out of cover", "h_b_w_stand_to_crouch_270_stop_003__A305"),
    ("Getting out of cover", "h_b_w_stand_to_crouch_270_stop_003__A304"),
    ("Getting out of cover", "h_b_w_stand_to_crouch_270_stop_002__A304"),
    ("Crouch start", "crouch_ff_start_225_R_001__A531"),
    ("Crouch start", "crouch_walk_forward_and_back_turn_360_to_idle_R_002__A493"),
    ("Crouch walk", "crouch_ff_loop_180_R_001__A531"),
)


def get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "GR00T-WholeBodyControl/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def validate_raw_csv(text: str, motion_name: str, expected_frames: int) -> None:
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        raise ValueError(f"{motion_name}: empty CSV")
    expected_root = [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
    ]
    if rows[0][:7] != expected_root:
        raise ValueError(f"{motion_name}: unexpected root columns: {rows[0][:7]}")
    if len(rows[0]) != 36 or len([c for c in rows[0] if c.endswith("_dof")]) != 29:
        raise ValueError(f"{motion_name}: expected 36 total columns and 29 joint DOFs")
    if len(rows) - 1 != expected_frames:
        raise ValueError(
            f"{motion_name}: metadata says {expected_frames} frames, CSV has {len(rows) - 1}"
        )
    if any(len(row) != 36 for row in rows[1:]):
        raise ValueError(f"{motion_name}: inconsistent row width")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/bones_seed_selected")
    )
    args = parser.parse_args()

    raw_dir = args.output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for category, motion_name in SELECTED_MOTIONS:
        quoted = urllib.parse.quote(motion_name, safe="")
        metadata = get_json(f"{VIEWER_API}/metadata/{quoted}")
        source_path = metadata.get("move_g1_path")
        if metadata.get("filename") != motion_name or not source_path:
            raise RuntimeError(f"{motion_name}: exact G1 path not found in metadata")

        payload = get_json(f"{VIEWER_API}/storage_local/g1csv/?csvpath={quoted}")
        if payload.get("name") != motion_name or "csv" not in payload:
            raise RuntimeError(f"{motion_name}: exact G1 CSV was not returned")
        validate_raw_csv(payload["csv"], motion_name, int(metadata["move_duration_frames"]))

        local_path = raw_dir / f"{motion_name}.csv"
        local_path.write_text(payload["csv"], encoding="utf-8")
        manifest_rows.append(
            {
                "category": category,
                "motion_name": motion_name,
                "source_path": source_path,
                "local_path": local_path.as_posix(),
            }
        )
        print(f"FOUND {motion_name}: {source_path}")

    manifest_path = args.output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("category", "motion_name", "source_path", "local_path")
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote {len(manifest_rows)} files and {manifest_path}")


if __name__ == "__main__":
    main()
