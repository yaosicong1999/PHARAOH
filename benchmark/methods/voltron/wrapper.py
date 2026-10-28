#!/usr/bin/env python3
"""VoltRon DAPI -> H&E registration wrapper.

Runs the installed VoltRon R package's own OME-TIFF importer (`VoltRon:::importImage`,
through RBioFormats) and its automated pairwise registration
(`VoltRon:::computeAutomatedPairwiseTransform`) -- the functions VoltRon's own
importImageData/registerSpatialData pipeline uses -- with SIFT + FLANN and a homography:

    query = DAPI (negated)   reference = H&E   matcher = FLANN   method = Homography

The pyramid level is chosen with the shared tier rule (largest level at or under the
tier's longest side) or pinned for the level-matched ablation. VoltRon returns a matrix
between the two imported images; it is rescaled to DAPI level 0 -> H&E level 0.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench import contract  # noqa: E402
from bench.pyramid import TIER_MAX_SIDE  # noqa: E402

# VoltRon's own GUI defaults. GOOD_MATCH_PERCENT and MAX_FEATURES are used by its ORB
# path only; FLANN/SIFT ignores them, but the routine requires them.
REGISTRATION = {"matcher": "FLANN", "method": "Homography",
                "good_match_percent": 0.20, "max_features": 1000,
                "negate_dapi": True, "negate_he": False}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dapi", required=True, type=Path)
    p.add_argument("--he", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--dapi-mpp", required=True, type=float)
    p.add_argument("--he-mpp", required=True, type=float)
    p.add_argument("--rscript", required=True, help="Rscript with VoltRon and RBioFormats")
    p.add_argument("--resolution", choices=tuple(TIER_MAX_SIDE))
    p.add_argument("--dapi-level", type=int, help="Pinned 0-based DAPI level (ablation).")
    p.add_argument("--he-level", type=int, help="Pinned 0-based H&E level (ablation).")
    return p


# --------------------------------------------------------------------------------------
# Pyramid levels as RBioFormats numbers them
# --------------------------------------------------------------------------------------
# tifffile reports a single level for a multi-file OME pyramid such as a Xenium
# morphology_focus_0000.ome.tif, so the SubIFD chain is also read directly from the
# TIFF/BigTIFF directories, and whichever reader finds more levels is used.

_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8,
               13: 4, 16: 8, 17: 8, 18: 8}
_INT_FORMATS = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 13: "I", 16: "Q", 17: "q", 18: "Q"}


def _ifd(path: Path, wanted: set, offset=None):
    """(endian, {tag: (type, count, raw)}, next_ifd) of one TIFF/BigTIFF directory."""
    with path.open("rb") as f:
        header = f.read(16)
        if header[:2] not in (b"II", b"MM"):
            raise ValueError(f"not a TIFF: {path}")
        endian = "<" if header[:2] == b"II" else ">"
        magic = struct.unpack(endian + "H", header[2:4])[0]
        if magic == 42:
            size, entry_size, count_fmt = 4, 12, "H"
            first = struct.unpack(endian + "I", header[4:8])[0]
        elif magic == 43:
            size, entry_size, count_fmt = 8, 20, "Q"
            first = struct.unpack(endian + "Q", header[8:16])[0]
        else:
            raise ValueError(f"unsupported TIFF version {magic}: {path}")
        f.seek(first if offset is None else offset)
        count = struct.unpack(endian + count_fmt, f.read(struct.calcsize(count_fmt)))[0]
        found = {}
        for _ in range(count):
            entry = f.read(entry_size)
            tag, kind = struct.unpack(endian + "HH", entry[:4])
            if size == 4:
                n, field = struct.unpack(endian + "I", entry[4:8])[0], entry[8:12]
            else:
                n, field = struct.unpack(endian + "Q", entry[4:12])[0], entry[12:20]
            if tag in wanted:
                found[tag] = (kind, n, field)
        next_ifd = struct.unpack(endian + ("I" if size == 4 else "Q"), f.read(size))[0]
        values = {}
        for tag, (kind, n, field) in found.items():
            nbytes = _TYPE_SIZES.get(kind, 0) * n
            if not nbytes:
                continue
            if nbytes <= size:
                values[tag] = (kind, n, field[:nbytes])
            else:
                f.seek(struct.unpack(endian + ("I" if size == 4 else "Q"), field)[0])
                values[tag] = (kind, n, f.read(nbytes))
        return endian, values, next_ifd


def _ints(endian, value) -> tuple:
    kind, n, raw = value
    fmt = _INT_FORMATS[kind]
    return struct.unpack(endian + f"{n}{fmt}", raw[: n * struct.calcsize(fmt)])


def _levels_stdlib(path: Path) -> list:
    endian, tags, next_ifd = _ifd(path, {254, 256, 257, 330})

    def wh(t):
        return _ints(endian, t[256])[0], _ints(endian, t[257])[0]
    levels = [wh(tags)]
    if 330 in tags:
        for offset in _ints(endian, tags[330]):
            if offset > 0:
                size = wh(_ifd(path, {256, 257}, offset)[1])
                if size not in levels:
                    levels.append(size)
        return levels
    seen = set()
    while next_ifd > 0 and next_ifd not in seen:
        seen.add(next_ifd)
        _, t, next_ifd = _ifd(path, {254, 256, 257}, next_ifd)
        size = wh(t)
        reduced = 254 in t and bool(_ints(endian, t[254])[0] & 1)
        if size == levels[0]:
            continue
        if not reduced and not (size[0] < levels[-1][0] and size[1] < levels[-1][1]):
            break
        if size not in levels:
            levels.append(size)
    return levels


def pyramid_levels(path: Path) -> list:
    """(width, height) per level, index 0 = RBioFormats resolution 1."""
    import tifffile
    found = []
    try:
        with tifffile.TiffFile(str(path)) as handle:
            series = handle.series[0]
            for level in getattr(series, "levels", None) or (series,):
                shape, axes = tuple(level.shape), getattr(level, "axes", "")
                if axes and "X" in axes and "Y" in axes:
                    found.append((int(shape[axes.index("X")]), int(shape[axes.index("Y")])))
                elif shape[-1] in (3, 4):
                    found.append((int(shape[-2]), int(shape[-3])))
                else:
                    found.append((int(shape[-1]), int(shape[-2])))
    except Exception:
        found = []
    if len(found) > 1:
        return found
    stdlib = _levels_stdlib(path)
    return stdlib if len(stdlib) >= len(found) else found


def select_level(levels: list, tier: str) -> int:
    cap = TIER_MAX_SIDE[tier]
    under = [i for i, (w, h) in enumerate(levels) if max(w, h) <= cap]
    return max(under, key=lambda i: max(levels[i])) if under else min(
        range(len(levels)), key=lambda i: max(levels[i]))


# --------------------------------------------------------------------------------------
# R
# --------------------------------------------------------------------------------------

def _q(value) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def r_script(args, outdir: Path, dapi_res: int, he_res: int, expected: dict) -> str:
    yes = lambda flag: '"Yes"' if flag else '"No"'  # noqa: E731
    reg = REGISTRATION
    return f'''
options(warn = 1)
elapsed <- function(t0) as.numeric(proc.time()[["elapsed"]] - t0)
t_total <- proc.time()[["elapsed"]]
suppressPackageStartupMessages(library(VoltRon))
suppressPackageStartupMessages(library(magick))
rbf <- tryCatch({{ loadNamespace("RBioFormats"); NULL }}, error = function(e) conditionMessage(e))
if (!is.null(rbf)) stop(sprintf(
  "RBioFormats could not be loaded (R %s, libPaths %s, JAVA_HOME %s): %s",
  R.version.string, paste(.libPaths(), collapse = ", "), Sys.getenv("JAVA_HOME", "(unset)"), rbf))
outdir <- {_q(outdir)}
message("[voltron] VoltRon ", as.character(utils::packageVersion("VoltRon")))

t0 <- proc.time()[["elapsed"]]
he <- VoltRon:::importImage(image = {_q(args.he.resolve())}, series = 1,
                            resolution = {he_res}, is.RGB = TRUE)[[1]]
he_import_sec <- elapsed(t0)
# channels = 1: DAPI is channel 0 of a Xenium morphology_focus, in file 0000. Leaving it
# NULL makes RBioFormats read all four channel files.
t0 <- proc.time()[["elapsed"]]
dapi <- VoltRon:::importImage(image = {_q(args.dapi.resolve())}, channels = 1, series = 1,
                              resolution = {dapi_res}, is.RGB = FALSE)[[1]]
dapi_import_sec <- elapsed(t0)
hi <- magick::image_info(he); di <- magick::image_info(dapi)
check <- function(label, info, w, h) {{
  if (abs(info$width[1] - w) > 1 || abs(info$height[1] - h) > 1)
    stop(sprintf("%s imported at %d x %d, selected level is %d x %d",
                 label, info$width[1], info$height[1], w, h))
}}
check("H&E", hi, {expected["he"][0]}, {expected["he"][1]})
check("DAPI", di, {expected["dapi"][0]}, {expected["dapi"][1]})

params <- list(
  automatictag = TRUE, GOOD_MATCH_PERCENT = {reg["good_match_percent"]},
  MAX_FEATURES = {reg["max_features"]}, Method = {_q(reg["method"])},
  Matcher = {_q(reg["matcher"])}, nonrigid = "None",
  channel_ref_image1 = "HE", channel_query_image1 = "HE",
  channel_ref_image2 = "DAPI", channel_query_image2 = "DAPI",
  scale_ref_image1 = 1, scale_query_image1 = 1, scale_ref_image2 = 1, scale_query_image2 = 1,
  negate_ref_image1 = {yes(reg["negate_he"])}, negate_query_image1 = {yes(reg["negate_he"])},
  negate_ref_image2 = {yes(reg["negate_dapi"])}, negate_query_image2 = {yes(reg["negate_dapi"])},
  flipflop_ref_image1 = "None", flipflop_query_image1 = "None",
  flipflop_ref_image2 = "None", flipflop_query_image2 = "None",
  rotate_ref_image1 = "0", rotate_query_image1 = "0",
  rotate_ref_image2 = "0", rotate_query_image2 = "0")

t0 <- proc.time()[["elapsed"]]
res <- VoltRon:::computeAutomatedPairwiseTransform(
  image_list = list(list(HE = he), list(DAPI = dapi)), channel_names = list("HE", "DAPI"),
  query_ind = 2, ref_ind = 1, input = params)
registration_sec <- elapsed(t0)
if (length(res$mapping) < 1 || length(res$mapping[[1]]) < 1) stop("VoltRon returned no mapping.")
H <- res$mapping[[1]][[1]]
if (is.null(H) || !(nrow(H) %in% c(2, 3))) stop("VoltRon returned no valid matrix.")
if (nrow(H) == 2) H <- rbind(H, c(0, 0, 1))
utils::write.table(H, file.path(outdir, "transform_imported.csv"), sep = ",",
                   row.names = FALSE, col.names = FALSE)
utils::write.table(data.frame(image = c("he", "dapi"), width = c(hi$width[1], di$width[1]),
                              height = c(hi$height[1], di$height[1])),
                   file.path(outdir, "dimensions.tsv"), sep = "\\t", row.names = FALSE, quote = FALSE)
if (inherits(res$alignment_image, "magick-image"))
  magick::image_write(res$alignment_image, file.path(outdir, "matches.png"), format = "png")
utils::write.table(data.frame(step = c("he_import", "dapi_import", "registration", "r_total"),
                              seconds = c(he_import_sec, dapi_import_sec, registration_sec, elapsed(t_total))),
                   file.path(outdir, "timings.tsv"), sep = "\\t", row.names = FALSE, quote = FALSE)
message("[voltron] done")
'''


def _archs(path: str):
    try:
        out = subprocess.run(["/usr/bin/lipo", "-archs", path], text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return set(out.stdout.split()) if out.returncode == 0 and out.stdout.strip() else None


def r_environment(rscript: str) -> dict:
    """Drop a JAVA_HOME whose libjvm cannot load into this R (e.g. an arm64 JDK beside an
    x86_64 R); rJava then finds a matching JVM on its own. Reported, never silent."""
    env = dict(os.environ)
    java = env.get("JAVA_HOME")
    if not java or not (Path(java) / "lib/server/libjvm.dylib").exists():
        return env
    r_binary = Path(rscript).resolve().parent / "exec" / "R"
    r_archs = _archs(str(r_binary if r_binary.exists() else Path(rscript).resolve()))
    jvm_archs = _archs(str(Path(java) / "lib/server/libjvm.dylib"))
    if r_archs and jvm_archs and not (r_archs & jvm_archs):
        print(f"[voltron] dropping JAVA_HOME={java} ({'/'.join(jvm_archs)}) for "
              f"R ({'/'.join(r_archs)})", flush=True)
        env.pop("JAVA_HOME")
    return env


def read_table(path: Path, delimiter: str) -> list:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def main() -> int:
    wrapper_start = time.perf_counter()
    args = build_parser().parse_args()
    pinned = args.dapi_level is not None and args.he_level is not None
    if pinned == (args.resolution is not None):
        raise SystemExit("pass exactly one of --resolution or (--dapi-level and --he-level)")
    for path in (args.dapi, args.he):
        if not path.exists():
            raise SystemExit(f"input not found: {path}")
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    levels = {"dapi": pyramid_levels(args.dapi), "he": pyramid_levels(args.he)}
    if pinned:
        chosen = {"dapi": args.dapi_level, "he": args.he_level}
    else:
        chosen = {name: select_level(sizes, args.resolution) for name, sizes in levels.items()}
    expected = {name: levels[name][chosen[name]] for name in levels}
    over_cap = ({name: max(expected[name]) > TIER_MAX_SIDE[args.resolution] for name in expected}
                if args.resolution else None)
    for name in ("dapi", "he"):
        print(f"[voltron] {name} level {chosen[name]}: {expected[name][0]}x{expected[name][1]}"
              + (" (no level under the tier; smallest used)" if over_cap and over_cap[name] else ""),
              flush=True)

    script = outdir / "_voltron.R"
    script.write_text(r_script(args, outdir, chosen["dapi"] + 1, chosen["he"] + 1, expected))
    r_start = time.perf_counter()
    proc = subprocess.run([args.rscript, str(script)], text=True, capture_output=True,
                          env=r_environment(args.rscript))
    r_wall_sec = time.perf_counter() - r_start
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    (outdir / "r_output.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        print(f"[voltron] R exited {proc.returncode}", file=sys.stderr)
        return proc.returncode

    import numpy as np
    H = np.loadtxt(outdir / "transform_imported.csv", delimiter=",")
    if H.shape == (2, 3):
        H = np.vstack([H, [0, 0, 1]])
    imported = {row["image"]: (int(float(row["width"])), int(float(row["height"])))
                for row in read_table(outdir / "dimensions.tsv", "\t")}
    frames = {name: contract.frame_from_pyramid(path, chosen[name], mpp, size_wh=imported[name])
              for name, path, mpp in (("dapi", args.dapi, args.dapi_mpp), ("he", args.he, args.he_mpp))}
    to_dapi_frame = np.diag([*(1 / np.asarray(contract.scale_xy(frames["dapi"]))), 1.0])
    to_he_level0 = np.diag([*contract.scale_xy(frames["he"]), 1.0])
    H0 = to_he_level0 @ H @ to_dapi_frame
    H0 /= H0[2, 2]
    contract.write_matrix(outdir, "VoltRon", frames, H0,
                          native={"imported_matrix": H.tolist(), "imported_wh": imported})

    r_timings = {f"r_{row['step']}_sec": float(row["seconds"])
                 for row in read_table(outdir / "timings.tsv", "\t")}
    timings = {**r_timings, "r_process_wall_sec": r_wall_sec,
               # The whole R process is charged: import through RBioFormats is VoltRon's.
               "method_sec": r_wall_sec,
               "wrapper_total_sec": time.perf_counter() - wrapper_start}
    (outdir / "run_metadata.json").write_text(json.dumps({
        "method": "voltron", "rscript": args.rscript, "registration": REGISTRATION,
        "dapi": str(args.dapi.resolve()), "he": str(args.he.resolve()),
        "resolution": args.resolution, "levels_0based": chosen,
        "pinned_levels": [args.dapi_level, args.he_level] if pinned else None,
        "level_wh": expected, "over_cap": over_cap, "imported_wh": imported,
        "level0_transform": H0.tolist(), "timings": timings,
    }, indent=2))
    print(f"[voltron] OK -> {outdir} | R {r_wall_sec:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
