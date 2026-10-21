#!/usr/bin/env python3
"""Build entry + repeated loop + exit into one SONIC reference motion."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--loop", type=Path, required=True)
    parser.add_argument("--loop-count", type=int, default=1)
    parser.add_argument("--exit", dest="exit_motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge-seconds", type=float, default=0.5)
    args = parser.parse_args()

    if args.loop_count < 1:
        parser.error("--loop-count must be at least 1")
    if args.bridge_seconds < 0:
        parser.error("--bridge-seconds must be non-negative")
    for motion in (args.entry, args.loop, args.exit_motion):
        if not motion.is_dir():
            parser.error(f"motion directory does not exist: {motion}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")

    sequence = [args.entry, *([args.loop] * args.loop_count), args.exit_motion]
    merger = Path(__file__).with_name("merge_sonic_reference_motions.py")

    with tempfile.TemporaryDirectory(prefix="sonic-sequence-") as temp_name:
        temp_root = Path(temp_name)
        current = sequence[0]
        for index, next_motion in enumerate(sequence[1:], start=1):
            is_last = index == len(sequence) - 1
            merged = args.output if is_last else temp_root / f"merge_{index:03d}"
            subprocess.run(
                [
                    sys.executable,
                    str(merger),
                    "--first",
                    str(current),
                    "--second",
                    str(next_motion),
                    "--output",
                    str(merged),
                    "--bridge-seconds",
                    str(args.bridge_seconds),
                ],
                check=True,
            )
            current = merged

    metadata = {
        "entry": args.entry.as_posix(),
        "loop": args.loop.as_posix(),
        "loop_count": args.loop_count,
        "exit": args.exit_motion.as_posix(),
        "bridge_seconds_each": args.bridge_seconds,
        "sequence": [motion.as_posix() for motion in sequence],
    }
    (args.output / "sequence_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
