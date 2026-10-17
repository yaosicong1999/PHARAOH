#!/usr/bin/env python3
"""Render viewable DAPI previews for run directories that already exist.

`1_dapi_level_image.png` is Stage 2's matching input, written from the raw 16-bit
DAPI with no windowing. That is why it looks almost black: a Xenium morphology image
fills only the bottom few percent of the uint16 range. Stage 1 now writes an 8-bit
`1_dapi_preview.png` beside it, but runs made before that do not have one, and
regenerating it does not require rerunning anything -- every intensity is already in
the file on disk.

Usage:
    python scripts/render_dapi_preview.py RUN_DIR [RUN_DIR ...]
    python scripts/render_dapi_preview.py outputs/parallel_runs_*/runs_*

A batch directory works too: any `runs_*` subdirectory below the given path is
picked up, so one path covers a whole parallel batch.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

LEVEL_IMAGE = "1_dapi_level_image.png"
PREVIEW_IMAGE = "1_dapi_preview.png"


def run_dirs(paths):
    """Expand each argument to the run directories it contains, or itself."""
    seen = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if (path / LEVEL_IMAGE).is_file():
            candidates = [path]
        else:
            candidates = sorted(
                child for child in path.glob("runs_*")
                if (child / LEVEL_IMAGE).is_file()
            )
        for candidate in candidates:
            if candidate not in seen:
                seen.append(candidate)
    return seen


def render(run_dir, *, low, high, overwrite):
    import cv2
    from my_utils import percentile_stretch_to_u8

    target = run_dir / PREVIEW_IMAGE
    if target.exists() and not overwrite:
        print(f"[preview] {run_dir.name}: {PREVIEW_IMAGE} exists (use --overwrite)")
        return False
    image = cv2.imread(str(run_dir / LEVEL_IMAGE), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"[preview] {run_dir.name}: could not read {LEVEL_IMAGE}", file=sys.stderr)
        return False
    preview = percentile_stretch_to_u8(image, low_percentile=low, high_percentile=high)
    if not cv2.imwrite(str(target), preview):
        print(f"[preview] {run_dir.name}: could not write {target}", file=sys.stderr)
        return False
    print(f"[preview] {run_dir.name}: wrote {PREVIEW_IMAGE} "
          f"({preview.shape[1]}x{preview.shape[0]}, {low}-{high} percentile window)")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render 1_dapi_preview.png from an existing run's Stage-1 image."
    )
    parser.add_argument("run_dirs", nargs="+",
                        help="run directories, or a batch directory holding runs_*")
    parser.add_argument("--low-percentile", type=float, default=1.0,
                        help="lower display bound (default: 1.0)")
    parser.add_argument("--high-percentile", type=float, default=99.5,
                        help="upper display bound (default: 99.5)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing preview")
    args = parser.parse_args(argv)

    if not 0.0 <= args.low_percentile < args.high_percentile <= 100.0:
        parser.error("need 0 <= --low-percentile < --high-percentile <= 100")

    targets = run_dirs(args.run_dirs)
    if not targets:
        parser.error(f"no run directory containing {LEVEL_IMAGE} in: "
                     f"{', '.join(args.run_dirs)}")
    written = sum(
        render(run_dir, low=args.low_percentile, high=args.high_percentile,
               overwrite=args.overwrite)
        for run_dir in targets
    )
    print(f"[preview] {written}/{len(targets)} run(s) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
