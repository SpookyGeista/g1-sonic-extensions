# G1 SONIC Extensions

VR teleoperation and motion-reference tools for the G1.

- Quest 3 body-retargeting corrections.
- Motion-reference conversion, trimming, and sequencing.
- ZMQ handover and simulation-playback extensions.

## Usage

Motion tools and tests are in `tools/`. Code changes are provided in `patches/`.
The base revision and application instructions are recorded in `UPSTREAM.json`.

```bash
python -m pytest -q tools/test_build_bones_continuous.py
```
