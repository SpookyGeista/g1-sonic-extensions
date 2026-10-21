# G1 SONIC Extensions

G1 tools for Quest 3 body retargeting, motion-reference processing, and switching between recorded motion capture and live full-body teleoperation.

## Operation guides

- [Motion capture → teleoperation → resume: three terminals, controls, and troubleshooting](docs/MOTION_CAPTURE_TELEOP.md)
- [Tool workflows and complete tool index](docs/TOOLS.md)

With `--soft-teleop-handover`, press **Enter** in the deployment terminal to pause a reference motion and switch to live POSE teleoperation. Press **Enter** again to blend back to the saved motion/frame. The guide covers calibration, stream readiness, and the required launch flags.

## Initialize the standalone tools

Python 3.11 is sufficient for the motion-processing tests:

```bash
git clone https://github.com/SpookyGeista/g1-sonic-extensions.git
cd g1-sonic-extensions
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-offline.txt
python -m pytest -q tools/test_build_bones_continuous.py
```

## Apply the integration changes

The patch targets the exact upstream revision recorded in `UPSTREAM.json`. From this repository:

```bash
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git ../GR00T-WholeBodyControl
cd ../GR00T-WholeBodyControl
git checkout c374bae5b9039cd0ee71377e654d11ce1bc69e1d
git submodule update --init --recursive
git apply --check ../g1-sonic-extensions/patches/local-code.patch
git apply ../g1-sonic-extensions/patches/local-code.patch
mkdir -p tools
cp -R ../g1-sonic-extensions/tools/. tools/
```

Complete the upstream environment and asset setup before running teleoperation or simulation. Apply the patch once to a clean checkout; keep the standalone environment separate from the Isaac environment.

## First motion operation

A prepared motion directory must contain the position and quaternion CSV channels expected by the tools. From either checkout containing `tools/`:

```bash
python tools/trim_sonic_reference_motion.py \
  --input /path/to/reference_motion \
  --output data/trimmed_motion --start-frame 0 --end-frame 99
```

The end frame is inclusive at 50 Hz. The tool writes a new motion directory and regenerates velocity channels. Use a fresh output directory for each operation.
