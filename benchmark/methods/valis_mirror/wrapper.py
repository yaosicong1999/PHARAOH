#!/usr/bin/env python3
"""VALIS with an option for a mirrored pair: the benchmark VALIS wrapper by default.

    python methods/valis_mirror/wrapper.py <VALIS wrapper arguments>
    python methods/valis_mirror/wrapper.py <...> --mirror double --he-flipped H_flipx
    python methods/valis_mirror/wrapper.py <...> --mirror valis

The VALIS wrapper arguments are those of methods/valis/wrapper.py (--dapi --he --outdir --dapi-mpp --he-mpp
--stage --caps-scale --dapi-reader ...; see its --help) and reach it unchanged.

--mirror off (default)
    methods/valis/wrapper.py itself: this process is replaced by it (os.execv) with the same arguments, so the
    output is exactly what it writes. VALIS does not test reflections (check_for_reflections=False).

--mirror double (method valis_double): the handedness is chosen outside VALIS.
    1. methods/valis/wrapper.py runs unchanged twice: the DAPI onto the H&E as given (RUN/as_given) and onto its
       left-right flipped copy (--he-flipped, RUN/flipped; checked to be that flip). One flip is enough: a flip
       in y is a flip in x plus a 180-degree rotation, which VALIS's own rotation estimate covers. A sub-run
       that fails counts as no match.
    2. Choice without ground truth: for each sub-run, the Pearson correlation, over the H&E's tissue (Otsu on
       the saturation of H&E level 5), between the H&E's hematoxylin (colour deconvolution) and the DAPI that
       the sub-run warps onto that H&E (level 5; the sub-run's point map, eval/common/pointmap.py, inverted:
       a matrix exactly, a field numerically). The higher one is kept; both values and their difference are
       recorded.
    3. RUN/double.json and a contract of kind "valis_double": the chosen sub-run's map (RUN/<chosen>, a
       contract of its own), then, if the flipped copy won, the flip back into the given H&E, the level-0
       matrix "he_flip_matrix" in double.json (x -> W - 1 - x).
       method_sec = both sub-runs + the choice.

--mirror valis (method valis_reflect): VALIS's own reflection search, check_for_reflections=True (its
    documented option): the rigid stage also tries the moving image flipped in y, in x and in both, and keeps
    the candidate with the smallest matched-feature residual. As shipped (valis-wsi 1.2.0, default DISK +
    LightGlue) it does not run, so two changes, recorded in run_metadata.json "variant", let it finish:
    - float32_workaround: matching a reflected candidate hands LightGlue float64 descriptors and keypoints next
      to float32 ones and torch stops ("expected m1 and m2 to have the same dtype, but got: double != float");
      LightGlueMatcher.match_images gets them cast to float32, a type change only (Catalign's benchmark
      patched the same).
    - fit_guard: a reflected candidate that finds (almost) no matches is fitted anyway and scikit-image's
      SimilarityTransform raises LinAlgError (SVD did not converge), even when the image is not mirrored.
      Only while align_to_prev_check_reflections runs, a fit that raises or is not finite leaves NaN
      parameters and its candidate an infinite error, so it can never be chosen; nothing else changes.
    run_metadata.json also gets "valis_reflection" (the sign of the exported rigid map's determinant:
    negative = VALIS applied a reflection) and "reflection_check": each candidate's error in VALIS's order
    (none, flip y, flip x, flip both = 180 deg), the fits that failed, and the candidate VALIS's argmin chose.
"""
import argparse
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # the folder that holds bench/, eval/ and methods/
WRAPPER = ROOT / "methods" / "valis" / "wrapper.py"
MODES = ("off", "double", "valis")


# --------------------------------------------------------------------------------------
# --mirror double
# --------------------------------------------------------------------------------------

LEVEL = 5


def check_flip(he: Path, flipped: Path) -> int:
    """Level-0 width; refuses a --he-flipped that is not the left-right flip of --he."""
    import tifffile
    with tifffile.TiffFile(he) as a, tifffile.TiffFile(flipped) as b:
        la, lb = a.series[0].levels, b.series[0].levels
        if [l.shape for l in la] != [l.shape for l in lb] or not np.array_equal(lb[-1].asarray(), la[-1].asarray()[:, ::-1]):
            raise SystemExit(f"{flipped} is not the left-right flip of {he}")
        return int(la[0].shape[1])


def projective(A: np.ndarray, X: np.ndarray, Y: np.ndarray) -> tuple:
    w = A[2, 0] * X + A[2, 1] * Y + A[2, 2]
    return (A[0, 0] * X + A[0, 1] * Y + A[0, 2]) / w, (A[1, 0] * X + A[1, 1] * Y + A[1, 2]) / w


class RunInverse:
    """H&E level-0 -> DAPI level-0 for one sub-run, consistent with its PointMap (DAPI -> H&E): a matrix
    inverted exactly; a field (valis_native) inverted numerically by chord iteration on the forward map."""

    def __init__(self, run_dir: Path):
        sys.path.insert(0, str(ROOT))
        from eval.common.pointmap import PointMap
        self.pm = PointMap(run_dir)
        d = self.pm.frames["dapi"]
        self.dapi_wh = (d["level0_width"], d["level0_height"])
        self.exact = None
        if self.pm.kind == "matrix":
            to_he = np.diag([*self.pm.he_scale, 1.0]) @ self.pm.matrix @ np.diag([*(1 / self.pm.dapi_scale), 1.0])
            self.exact = np.linalg.inv(to_he)
        else:
            # global affine H&E -> DAPI from a coarse forward grid, the chord iteration's Jacobian
            w, h = self.dapi_wh
            gx, gy = np.meshgrid(np.linspace(0, w - 1, 120), np.linspace(0, h - 1, 120))
            src = np.column_stack([gx.ravel(), gy.ravel()])
            dst, ok, _ = self.pm.apply(src)
            A, *_ = np.linalg.lstsq(np.column_stack([dst[ok], np.ones(ok.sum())]), src[ok], rcond=None)
            self.coarse = A.T

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> tuple:
        if self.exact is not None:
            return projective(self.exact, X, Y)
        from scipy.interpolate import RegularGridInterpolator
        step = 4                                                  # control points every 4 output px, plus the last
        rows = np.unique(np.r_[np.arange(0, X.shape[0], step), X.shape[0] - 1])
        cols = np.unique(np.r_[np.arange(0, X.shape[1], step), X.shape[1] - 1])
        target = np.column_stack([X[np.ix_(rows, cols)].ravel(), Y[np.ix_(rows, cols)].ravel()])
        # chord iteration on the run's forward map: u <- u + J^-1 (x - F(u)), J^-1 from the global affine
        u = np.column_stack([target, np.ones(len(target))]) @ self.coarse.T
        for _ in range(30):
            f, ok, _ = self.pm.apply(u)
            r = target - f
            u = u + r @ self.coarse[:, :2].T
            if not ok.any() or np.nanmax(np.abs(r[ok])) < 1e-3:
                break
        f, ok, _ = self.pm.apply(u)
        ok &= np.hypot(*(target - f).T) < 0.05                    # a converged root, in H&E level-0 px
        u[~ok] = np.nan
        grid = (rows.astype(float), cols.astype(float))
        rr, cc = np.meshgrid(np.arange(X.shape[0], dtype=float), np.arange(X.shape[1], dtype=float), indexing="ij")
        q = np.column_stack([rr.ravel(), cc.ravel()])
        out = [RegularGridInterpolator(grid, u[:, k].reshape(len(rows), len(cols)))(q).reshape(X.shape) for k in (0, 1)]
        return out[0], out[1]


def level_grid(shape0, shape_k):
    """H&E level-0 coordinates of the pixel centres of level k."""
    fy, fx = shape0[0] / shape_k[0], shape0[1] / shape_k[1]
    return np.meshgrid(fx * (np.arange(shape_k[1]) + 0.5) - 0.5, fy * (np.arange(shape_k[0]) + 0.5) - 0.5)


def sample_dapi(path: Path, U: np.ndarray, V: np.ndarray, level: int) -> np.ndarray:
    """Bilinear DAPI values at DAPI level-0 coordinates (U, V), read from `level` (or the coarsest level if the
    pyramid has fewer); 0 outside. Read with bench.pyramid, which also finds the pyramid of a Xenium multi-file
    morphology_focus_0000.ome.tif (its reduced levels are SubIFDs that tifffile's series does not list)."""
    import cv2
    from bench.pyramid import level_sizes, read_level
    sizes = level_sizes(path)
    level = min(level, len(sizes) - 1)
    (w0, h0), (wk, hk) = sizes[0], sizes[level]
    xk = (U + 0.5) * wk / w0 - 0.5
    yk = (V + 0.5) * hk / h0 - 0.5
    finite = np.isfinite(xk) & np.isfinite(yk)
    if not finite.any():
        return np.zeros(U.shape, np.float32)
    xa, xb = max(int(np.floor(xk[finite].min())) - 2, 0), min(int(np.ceil(xk[finite].max())) + 3, wk)
    ya, yb = max(int(np.floor(yk[finite].min())) - 2, 0), min(int(np.ceil(yk[finite].max())) + 3, hk)
    if xa >= xb or ya >= yb:
        return np.zeros(U.shape, np.float32)
    image = read_level(path, level)
    if image.ndim != 2:
        raise SystemExit(f"{path}: level {level} has shape {image.shape}; expected one DAPI channel")
    crop = np.asarray(image[ya:yb, xa:xb]).astype(np.float32)
    mx = np.where(finite, xk - xa, -10).astype(np.float32)
    my = np.where(finite, yk - ya, -10).astype(np.float32)
    return cv2.remap(crop, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def similarity(run_dir: Path, he: Path, dapi: Path) -> float:
    """Hematoxylin vs warped DAPI correlation over the H&E tissue at LEVEL."""
    sys.path.insert(0, str(ROOT))
    import cv2
    from scipy import ndimage
    from skimage import filters
    from skimage.color import rgb2hed
    from bench.pyramid import level_sizes, read_level
    sizes = level_sizes(he)
    shape0 = sizes[0][::-1]                                       # (height, width)
    rgb = read_level(he, min(LEVEL, len(sizes) - 1))
    sat = ndimage.gaussian_filter(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32), 2)
    tissue = sat > filters.threshold_otsu(sat)
    X, Y = level_grid(shape0, rgb.shape[:2])
    warped = sample_dapi(dapi, *RunInverse(run_dir)(X, Y), LEVEL)
    hema = rgb2hed(rgb)[..., 0]
    return float(np.corrcoef(hema[tissue], warped[tissue])[0, 1])


def run_double(passthrough: list, he_flipped: Path) -> int:
    p = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("--dapi", "--he", "--outdir"):
        p.add_argument(name, required=True, type=Path)
    p.add_argument("--dapi-mpp", required=True)
    p.add_argument("--he-mpp", required=True)
    args, rest = p.parse_known_args(passthrough)
    start = time.perf_counter()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    width = check_flip(args.he, he_flipped)
    subruns = {}
    for name, he in (("as_given", args.he), ("flipped", he_flipped)):
        sub = outdir / name
        sub.mkdir(exist_ok=True)
        command = [sys.executable, str(WRAPPER), "--dapi", str(args.dapi), "--he", str(he), "--outdir", str(sub),
                   "--dapi-mpp", args.dapi_mpp, "--he-mpp", args.he_mpp, *rest]
        t0 = time.perf_counter()
        with open(sub / "run.log", "w") as log:
            log.write(" ".join(command) + "\n\n")
            log.flush()
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        ok = code == 0 and (sub / "run_metadata.json").exists() and (sub / "coordinate_contract.json").exists()
        info = {"he": str(Path(he).resolve()), "returncode": code, "outcome": "success" if ok else "registration_error",
                "wall_sec": time.perf_counter() - t0}
        if ok:
            info["method_sec"] = json.loads((sub / "run_metadata.json").read_text())["timings"]["method_sec"]
            (sub / "run_record.json").write_text(json.dumps({"outcome": "success", "role": "valis_double sub-run",
                                                             "method_sec": info["method_sec"]}, indent=2))
        subruns[name] = info
        print(f"[valis_double] {name}: {info['outcome']} ({info['wall_sec']:.0f} s)", flush=True)
    t0 = time.perf_counter()
    for name, info in subruns.items():
        if info["outcome"] == "success":
            info["similarity"] = similarity(outdir / name, Path(info["he"]), args.dapi)
            print(f"[valis_double] {name}: hematoxylin-DAPI correlation {info['similarity']:.4f}", flush=True)
    scored = {n: i["similarity"] for n, i in subruns.items() if "similarity" in i and np.isfinite(i["similarity"])}
    if not scored:
        raise SystemExit("neither sub-run produced a usable registration")
    chosen = max(scored, key=scored.get)
    choice_sec = time.perf_counter() - t0
    flip = [[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    (outdir / "double.json").write_text(json.dumps({
        "chosen": chosen, "similarity": scored,
        "margin": scored[chosen] - min(scored.values()) if len(scored) == 2 else None,
        "he_flip_matrix": flip, "criterion": f"Pearson r, H&E hematoxylin vs warped Hoechst over H&E tissue, level {LEVEL}",
        "subruns": subruns}, indent=2))
    sys.path.insert(0, str(ROOT))
    from bench import contract
    frames = json.loads((outdir / chosen / "coordinate_contract.json").read_text())["frames"]
    frames["he"] = dict(frames["he"], image_path=str(args.he.resolve()), chosen_input=subruns[chosen]["he"])
    contract.write(outdir, "VALIS (two runs: H&E as given and flipped)", frames,
                   {"kind": "valis_double", "path": "double.json", "stored_direction": "dapi_to_he", "requires_inverse": False},
                   native={"chosen": chosen, "similarity": scored})
    method_sec = sum(i.get("method_sec", 0.0) for i in subruns.values()) + choice_sec
    (outdir / "run_metadata.json").write_text(json.dumps({
        "method": "valis_double", "wrapper": str(Path(__file__).resolve()), "mirror": "double",
        "benchmark_wrapper": str(WRAPPER), "passthrough_args": rest, "chosen": chosen, "similarity": scored,
        "subruns": subruns,
        "timings": {"method_sec": method_sec, "choice_sec": choice_sec, "wrapper_total_sec": time.perf_counter() - start},
    }, indent=2))
    print(f"[valis_double] OK -> {outdir} | chose {chosen} | method {method_sec:.1f}s", flush=True)
    return 0


# --------------------------------------------------------------------------------------
# --mirror valis
# --------------------------------------------------------------------------------------

CANDIDATES = ("none", "flip y (top-bottom)", "flip x (left-right)", "flip x and y (= 180 deg)")
CHECKS = []          # one entry per reflection check VALIS ran (one per image pair)


def guard_fits() -> None:
    """Make a failed fit of a reflected candidate count as no match, and log every candidate's error."""
    from valis import serial_rigid, warp_tools
    check = serial_rigid.SerialRigidRegistrar.align_to_prev_check_reflections
    measure = warp_tools.measure_error

    def guarded(self, *args, **kwargs):
        transformer = kwargs["transformer"] if "transformer" in kwargs else args[0]
        estimate = transformer.estimate
        log = {"errors": [], "failed_fits": []}
        CHECKS.append(log)

        def safe_estimate(src, dst, *a, **k):
            fit = len(log["errors"])
            try:
                ok = estimate(src, dst, *a, **k)
            except (np.linalg.LinAlgError, ValueError) as error:
                ok, reason = False, f"{type(error).__name__}: {error}"
            else:
                reason = None if ok is not False and np.isfinite(transformer.params).all() else "no finite fit"
            if reason:
                transformer.params = np.full((3, 3), np.nan)
                log["failed_fits"].append({"candidate": CANDIDATES[fit % 4], "n_matches": int(len(src)), "reason": reason})
                return False
            return ok

        def safe_measure(*a, **k):
            try:
                tre, d = measure(*a, **k)
            except (ValueError, ZeroDivisionError):
                tre, d = np.nan, np.nan
            d = float(d) if np.isfinite(d) else np.inf
            log["errors"].append(d)
            return (tre if np.isfinite(tre) else np.inf), d

        transformer.estimate = safe_estimate
        warp_tools.measure_error = safe_measure
        try:
            return check(self, *args, **kwargs)
        finally:
            del transformer.estimate
            warp_tools.measure_error = measure

    serial_rigid.SerialRigidRegistrar.align_to_prev_check_reflections = guarded


def run_valis_check(passthrough: list) -> int:
    from valis import registration
    shipped = registration.Valis.__init__

    def with_reflections(self, *args, **kwargs):
        kwargs["check_for_reflections"] = True
        return shipped(self, *args, **kwargs)

    registration.Valis.__init__ = with_reflections

    from valis import feature_matcher
    lightglue = feature_matcher.LightGlueMatcher.match_images

    def float32(self, *args, **kwargs):
        for key in ("desc1", "desc2", "kp1_xy", "kp2_xy"):
            if isinstance(kwargs.get(key), np.ndarray):
                kwargs[key] = kwargs[key].astype(np.float32)
        return lightglue(self, *args, **kwargs)

    feature_matcher.LightGlueMatcher.match_images = float32
    guard_fits()
    outdir = Path(passthrough[passthrough.index("--outdir") + 1])
    sys.argv = [str(WRAPPER), *passthrough]
    try:
        runpy.run_path(str(WRAPPER), run_name="__main__")
        code = 0
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
    if code == 0:
        metadata = outdir / "run_metadata.json"
        payload = json.loads(metadata.read_text())
        rigid = outdir / "transform_level0.csv"
        if rigid.exists():
            M = np.loadtxt(rigid, delimiter=",")
        else:
            M = np.asarray(json.loads((outdir / "valis_native" / "transform.json").read_text())["slides"]["dapi"]["M"])
        payload["variant"] = {"check_for_reflections": True, "float32_workaround": True, "fit_guard": True,
                              "wrapper": str(Path(__file__).resolve()), "mirror": "valis",
                              "benchmark_wrapper": str(WRAPPER)}
        payload["valis_reflection"] = bool(np.linalg.det(M[:2, :2]) < 0)
        payload["reflection_check"] = [
            {"candidates": list(CANDIDATES[:len(c["errors"])]),
             "errors": [e if np.isfinite(e) else "inf" for e in c["errors"]],
             "failed_fits": c["failed_fits"],
             "chosen": CANDIDATES[int(np.argmin(c["errors"]))] if c["errors"] else None} for c in CHECKS]
        metadata.write_text(json.dumps(payload, indent=2))
        for c in payload["reflection_check"]:
            print(f"[valis_reflect] candidates {dict(zip(c['candidates'], c['errors']))}; failed fits "
                  f"{[f['candidate'] for f in c['failed_fits']]}; chosen: {c['chosen']}", flush=True)
        print(f"[valis_reflect] VALIS {'applied' if payload['valis_reflection'] else 'did not apply'} a reflection", flush=True)
    return code


# --------------------------------------------------------------------------------------

def main() -> int:
    # allow_abbrev=False: otherwise argparse would take the benchmark's --he for an abbreviation of --he-flipped
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--mirror", choices=MODES, default="off",
                   help="off (default): methods/valis/wrapper.py as is; double: two runs, H&E as given and flipped, "
                        "the better kept; valis: VALIS's own check_for_reflections")
    p.add_argument("--he-flipped", type=Path, help="--mirror double: the left-right flipped copy of --he")
    args, passthrough = p.parse_known_args()
    if (args.he_flipped is not None) != (args.mirror == "double"):
        p.error("--he-flipped goes with --mirror double, and only with it")
    if args.mirror == "off":
        os.execv(sys.executable, [sys.executable, str(WRAPPER), *passthrough])
    if args.mirror == "double":
        return run_double(passthrough, args.he_flipped)
    return run_valis_check(passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
