#!/usr/bin/env python3
"""PHARAOH DAPI -> H&E registration wrapper.

Runs `run_pharaoh.py` from the frozen snapshot beside this file (version_4.15/, recorded
in SNAPSHOT.json) as one automatic, unattended run: no manual Stage-2 fallback, no
failsafe GUI, no QC images, no internal monitor (the runner samples memory). PHARAOH
selects its own pyramid levels and emits one DAPI->H&E level-0 homography -- whatever
Stage 8's gate promoted, never the rejected candidate -- so there is one condition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench import contract  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dapi", required=True, type=Path)
    p.add_argument("--he", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path, help="PHARAOH's own run directory")
    p.add_argument("--dapi-mpp", required=True, type=float)
    p.add_argument("--he-mpp", type=float, default=None,
                   help="H&E level-0 µm per pixel; recorded only (PHARAOH works in pixels); omit when unknown")
    p.add_argument("--pharaoh-root", required=True, type=Path,
                   help="Version directory holding run_pharaoh.py")
    return p


def promoted_matrix(run_dir: Path):
    """Stage 8's promoted level-0 matrix; the JSON keeps full precision."""
    import numpy as np
    path = run_dir / "dapi_to_he_homography_level0.json"
    if path.exists():
        payload = json.loads(path.read_text())
        for key in ("homography_3x3", "matrix"):
            if key in payload:
                return np.asarray(payload[key], dtype=float)
    path = run_dir / "dapi_to_he_homography_level0.csv"
    return np.loadtxt(path, delimiter=",") if path.exists() else None


def main() -> int:
    args = build_parser().parse_args()
    wrapper_start = time.perf_counter()
    script = args.pharaoh_root / "run_pharaoh.py"
    for path in (args.dapi, args.he, script):
        if not path.exists():
            raise SystemExit(f"not found: {path}")
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script.resolve()), "--pair", "dapi-he",
               "--moving", str(args.dapi.resolve()), "--fixed", str(args.he.resolve()),
               "--run-dir", str(outdir), "--force",
               "--stage2-nomanual", "--no-failsafe", "--no-qc-images", "--no-monitor"]
    print("[pharaoh] " + " ".join(command), flush=True)
    method_start = time.perf_counter()
    # cwd is the version directory so PHARAOH's own relative imports resolve.
    completed = subprocess.run(command, cwd=str(args.pharaoh_root.resolve()))
    method_sec = time.perf_counter() - method_start
    if completed.returncode != 0:
        raise SystemExit(f"run_pharaoh.py exited {completed.returncode}")

    import numpy as np
    matrix = promoted_matrix(outdir)
    if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise SystemExit("PHARAOH produced no finite 3x3 level-0 transform")
    info = json.loads((outdir / "images_info.json").read_text())
    frames = {"dapi": contract.frame_from_pyramid(args.dapi, int(info["DAPI_level"]), args.dapi_mpp),
              "he": contract.frame_from_pyramid(args.he, int(info["HE_level"]), args.he_mpp)}
    contract.write_matrix(outdir, "PHARAOH", frames, matrix,
                          native={"output": "Stage 8 promoted level-0 matrix",
                                  "images_info": info})
    stage_times = {}
    if (outdir / "pipeline_times.json").exists():
        stage_times = {k: v for k, v in json.loads((outdir / "pipeline_times.json").read_text()).items()
                       if isinstance(v, (int, float))}
    root = args.pharaoh_root.resolve()
    (outdir / "run_metadata.json").write_text(json.dumps({
        "method": "pharaoh", "pharaoh_root": str(root), "pharaoh_version": root.name,
        "command": command, "python": sys.executable, "device": "cpu",
        "run_pharaoh_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "parameters_sha256": hashlib.sha256((root / "scripts/parameters.json").read_bytes()).hexdigest(),
        "dapi": str(args.dapi.resolve()), "he": str(args.he.resolve()),
        "levels_0based": {"dapi": int(info["DAPI_level"]), "he": int(info["HE_level"])},
        "level0_transform": matrix.tolist(),
        "timings": {"method_sec": method_sec, "pipeline_stage_sec": stage_times,
                    "wrapper_total_sec": time.perf_counter() - wrapper_start},
    }, indent=2))
    print(f"[pharaoh] OK -> {outdir} | method {method_sec:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
