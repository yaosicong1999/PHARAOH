"""Transcript agreement with 10x's alignment matrix, in H&E micrometres, for every method.

Every Xenium transcript is scored, read from the dataset's transcripts.parquet. Its position is
in micrometres in the DAPI frame (x_location, y_location); dividing by dapi_mpp (um per level-0 pixel) gives the DAPI
level-0 point. 10x's matrix maps H&E level 0 -> DAPI level 0, so its inverse gives the
reference H&E position. The run maps the same DAPI point; the distance between the two,
in H&E level-0 pixels times he_mpp, is the disagreement. Every method is measured this
way, RegWSI included (its pull field is inverted per point, eval/common/regwsi_inverse.py).

This is agreement with 10x, not accuracy: 10x's matrix is another method's output.
Every transcript the run maps is measured, including one mapped outside the H&E image
(counted as n_outside_he); one it cannot map is excluded and counted as n_invalid.

Each dataset's transcript positions are read once from the Xenium output and cached as
<cache_root>/transcripts/<dataset>/xy_um_<key>.npy (source dtype, file order); a run reads
that array in chunks (configs/eval.json, chunk_size).

A result is reused, not recomputed, when the same transform was already scored: results
are also stored under results/_by_artifact/matrix/<method>/<key>.json, keyed by the
content of the run's transform files and coordinate frames, the dataset's transcripts and
the scoring code. Repeated series whose registrations are byte-identical therefore cost
only the hashing.

Writes results/<series>/matrix/<method>/
    runs/<dataset>/<condition>.json      statistics, invalid reasons, histogram
    runs.csv                             one row per run, failures included
    summary.json
"""
from __future__ import annotations

import gc
import hashlib
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from bench import contract
from bench.config import CONFIGS, environment, read_json
from .pointmap import PointMap
from .runs import (SCORING_CODE_SHA256, METHOD_KINDS, cached_result, distance_stats, dump, iter_runs, method_root, parse_args,
                   peak_memory, results_dir, run_key, write_table)

HISTOGRAM_EDGES_UM = np.concatenate([np.arange(0, 20, 0.1), np.arange(20, 100, 1), [100, 1000, np.inf]])


def iter_transcripts_um(path: Path, chunk: int):
    """(N, 2) x/y micrometre chunks of a Xenium transcripts.parquet, in file order and the source
    dtype. Read row group by row group, so every row is returned even where an old writer stored
    a wrong file-level row count (the breast-cancer replicates s1r1 / s1r2)."""
    if Path(path).suffix != ".parquet":
        raise ValueError(f"{path}: transcripts are read from transcripts.parquet only")
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk, columns=["x_location", "y_location"]):
        yield np.column_stack([batch.column(0).to_numpy(), batch.column(1).to_numpy()])


def file_key(path: Path) -> str:
    stat = path.resolve().stat()
    return hashlib.sha256(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:16]


def transcripts_um(entry: dict, chunk: int) -> tuple[np.ndarray, str]:
    """Every transcript's (x, y) in micrometres, read once and cached; and the cache key."""
    source = Path(entry["transcripts"])
    key = file_key(source)
    cache = Path(environment()["cache_root"]) / "transcripts" / entry["name"] / f"xy_um_{key}.npy"
    if not cache.exists():
        start = time.perf_counter()
        parts = list(iter_transcripts_um(source, chunk))
        xy = np.concatenate(parts) if parts else np.empty((0, 2), np.float32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        partial = cache.with_suffix(".partial.npy")
        np.save(partial, xy)
        partial.replace(cache)
        cache.with_suffix(".json").write_text(json.dumps(
            {"source": str(source.resolve()), "n": int(len(xy)), "dtype": str(xy.dtype),
             "units": "micrometre, DAPI frame", "read_sec": time.perf_counter() - start}) + "\n")
        print(f"[matrix] {entry['name']}: cached {len(xy):,} transcripts ({time.perf_counter() - start:.0f}s)",
              flush=True)
    return np.load(cache, mmap_mode="r"), key


def reference_dapi_to_he(entry: dict) -> np.ndarray:
    alignment = entry["alignment"]
    matrix = np.loadtxt(alignment["path"], delimiter=",")
    if matrix.shape != (3, 3) or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError(f"{entry['name']}: 10x matrix is not a nonsingular 3x3")
    if alignment.get("direction") == "he_level0_px_to_dapi_level0_px":
        return np.linalg.inv(matrix)
    if alignment.get("direction") == "dapi_level0_px_to_he_level0_px":
        return matrix
    raise ValueError(f"{entry['name']}: unknown 10x matrix direction")


# --------------------------------------------------------------------------------------
# Content keys: identical transforms share one result
# --------------------------------------------------------------------------------------

def _digest(path: Path, memo: dict) -> str:
    """sha256 of a file, remembered by (path, size, mtime) so it is read once."""
    stat = path.stat()
    known = memo.get(str(path))
    if known and known[0] == stat.st_size and known[1] == stat.st_mtime_ns:
        return known[2]
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            h.update(block)
    memo[str(path)] = [stat.st_size, stat.st_mtime_ns, h.hexdigest()]
    return memo[str(path)][2]


def artifact_key(run_dir: Path, dataset: str, transcripts_key: str, memo: dict) -> str:
    """The run's transform (its files' content and coordinate frames), the dataset and its
    transcripts, and the scoring code."""
    c = contract.read(run_dir)
    artifact = c["artifact"]
    files = [run_dir / artifact["path"]]
    if artifact.get("postprocessing_path"):
        files.append(run_dir / artifact["postprocessing_path"])
    if artifact["kind"] == "valis_native":
        files += sorted((run_dir / artifact["path"]).parent.glob("*.npy"))
    parts = [dataset, transcripts_key, SCORING_CODE_SHA256, artifact["kind"],
             json.dumps(c["frames"], sort_keys=True)]
    parts += [f"{f.relative_to(run_dir)}:{_digest(f, memo)}" for f in files]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------

def score(entry: dict, run_dir: Path, method: str, xy_um: np.ndarray, chunk: int) -> dict:
    pm = PointMap(run_dir)
    if pm.kind not in METHOD_KINDS[method]:
        raise ValueError(f"{pm.kind} is not a {method} artifact")
    reference = reference_dapi_to_he(entry)
    dapi_mpp, he_mpp = float(entry["dapi_mpp"]), float(entry["he_mpp"])
    n = len(xy_um)
    distance = np.empty(n)
    reasons = Counter()
    for start in range(0, n, chunk):
        dapi = np.asarray(xy_um[start:start + chunk], dtype=np.float64) / dapi_mpp  # um -> DAPI level-0 px
        hom = np.column_stack([dapi, np.ones(len(dapi))]) @ reference.T
        he = hom[:, :2] / hom[:, 2:]
        predicted, valid, reason = pm.apply(dapi)
        d = np.linalg.norm(predicted - he, axis=1) * he_mpp
        d[~valid] = np.nan
        distance[start:start + len(d)] = d
        reasons.update(reason.tolist())
    finite = distance[np.isfinite(distance)]
    counts, _ = np.histogram(finite, bins=HISTOGRAM_EDGES_UM)
    result = {"kind": pm.kind, "n_transcripts_total": int(n), "n_sampled": int(n),
              "n_invalid": int(n - len(finite)), "n_outside_he": int(reasons.get("target_outside_he", 0)),
              **distance_stats(finite),
              "invalid_reasons": {k: v for k, v in reasons.items() if k not in ("ok", "target_outside_he")},
              "point_map": pm.describe(),
              "histogram": {"edges_um": [float(e) if np.isfinite(e) else None for e in HISTOGRAM_EDGES_UM],
                            "counts": counts.tolist()}}
    del pm, distance, finite
    return result


def evaluate(series: str, method: str, args, settings: dict) -> None:
    root = method_root(series, "matrix", method)
    if not root.is_dir():
        print(f"[matrix] {series} {method}: no runs under {root}; skipped", flush=True)
        return
    out = results_dir(series, "matrix", method)
    store = Path(environment()["results_root"]) / "_by_artifact" / "matrix" / method
    memo_path = Path(environment()["results_root"]) / "_by_artifact" / "file_digests.json"
    memo = read_json(memo_path) if memo_path.exists() else {}
    chunk = int(settings["chunk_size"])
    rows, loaded = [], {}
    for entry, condition, run_dir, record, selected in iter_runs(series, "matrix", method,
                                                                 args.datasets, args.conditions):
        row = {"dataset": entry["name"], "condition": condition, "outcome": record["outcome"],
               "reason": record.get("reason"), "method_sec": record.get("method_sec"),
               **peak_memory(run_dir, record)}
        if record["outcome"] != "success":
            rows.append(row)
            continue
        result_path = out / "runs" / entry["name"] / f"{condition}.json"
        result = cached_result(result_path, run_dir, args.rescore and selected)
        source = "cached" if result is not None else None
        if result is None and not selected:  # outside --datasets/--conditions: keep the table whole
            rows.append({**row, "eval_error": "not scored yet"})
            continue
        started = time.perf_counter()
        if result is None:
            try:
                if entry["name"] not in loaded:
                    loaded.clear()  # one dataset's transcripts in memory at a time
                    gc.collect()
                    loaded[entry["name"]] = transcripts_um(entry, chunk)
                xy_um, transcripts_key = loaded[entry["name"]]
                key = artifact_key(run_dir, entry["name"], transcripts_key, memo)
                dump(memo_path, memo)
                shared = store / f"{key}.json"
                if shared.exists() and not args.rescore:
                    result, source = read_json(shared), "same transform scored before"
                else:
                    result = {"artifact_key": key, **score(entry, run_dir, method, xy_um, chunk)}
                    dump(shared, result)
                result = {**result, "scoring_code_sha256": SCORING_CODE_SHA256,
                          "run_record_sha256": run_key(run_dir)}
                dump(result_path, result)
            except Exception as exc:
                result = {"eval_error": f"{type(exc).__name__}: {exc}"}
            gc.collect()
        row.update({k: v for k, v in result.items()
                    if k not in ("scoring_code_sha256", "run_record_sha256", "artifact_key",
                                 "histogram", "invalid_reasons", "point_map")})
        rows.append(row)
        if selected:
            seconds = time.perf_counter() - started
            line = (f"median {row['median_um']:.2f} um ({row['n_scored']:,}/{row['n_sampled']:,}"
                    + (f", {row['n_outside_he']:,} outside H&E" if row.get("n_outside_he") else "") + ")"
                    if row.get("median_um") is not None else row.get("eval_error") or row["outcome"])
            print(f"[matrix] {series} {method} {condition:16s} {entry['name']}: {line}"
                  + (f" [{source}]" if source else f" {seconds:.0f}s"), flush=True)
    write_table(out / "runs.csv", rows)
    dump(out / "summary.json", {"method": method, "series": series, "mode": "matrix",
                                "scoring_code_sha256": SCORING_CODE_SHA256,
                                "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                                "metric": "distance to 10x's matrix position, H&E level-0 micrometres, "
                                          "every transcript", "settings": settings, "runs": rows})
    print(f"[matrix] {series}: {len(rows)} runs -> {out}", flush=True)


def main(method: str) -> int:
    args = parse_args("matrix", method)
    settings = read_json(CONFIGS / "eval.json")["matrix"]
    for series in args.series:
        evaluate(series, method, args, settings)
    return 0
