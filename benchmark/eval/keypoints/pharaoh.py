#!/usr/bin/env python3
"""Keypoint TRE (10x landmarks, H&E micrometres) for PHARAOH runs.

    <eval_python.pharaoh> eval/keypoints/pharaoh.py --series iter1 [iter2 ...] [--datasets ...] [--conditions ...]

<eval_python.pharaoh> is the interpreter configs/environment.json names for this method.

Logic: eval/common/keypoints.py. Output: results/<series>/keypoints/pharaoh/.
"""
import sys
from pathlib import Path

# Replace, not prepend: this file's own folder must not stay on the path, or a script
# named after a package would shadow that package.
sys.path[0] = str(Path(__file__).resolve().parents[2])
from eval.common.keypoints import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("pharaoh"))
