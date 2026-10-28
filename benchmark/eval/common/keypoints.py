"""Keypoint TRE: 10x's own landmarks, DAPI side mapped by the run, error in H&E micrometres.

    TRE_i = |map(dapi_i) - he_i| * he_mpp        (H&E level-0 pixels -> um)

Landmarks are read from the path each manifest entry declares, in either shape 10x ships:
a four-column CSV (fixedX/fixedY = DAPI level 0, alignmentX/alignmentY = H&E level 0), or
two headerless TSVs paired by row order, each scaled to level 0 by the factor the manifest
states.

Every landmark the run maps is measured, including one mapped outside the H&E image (a bad
registration, not a missing value; counted as n_outside_he). A landmark the run cannot map
(outside the method's field domain, no unique inverse root) stays in the table with NaN and
a reason; it is excluded from the statistics and counted as n_unmeasured, and the run is
then not `complete`. Only complete runs compete for a dataset's best condition, and
`matched` (an ablation) never does.

Writes results/<series>/keypoints/<method>/
    points/<dataset>/<condition>.csv     every landmark
    points/<dataset>/<condition>.json    the run's statistics (reused until its run is redone)
    runs.csv                             one row per run, failures included
    summary.json                         every run, plus the best condition per dataset
"""
from __future__ import annotations

import csv
import gc
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from .pointmap import PointMap
from .runs import (SCORING_CODE_SHA256, METHOD_KINDS, cached_result, distance_stats, dump, iter_runs, method_root, parse_args,
                   peak_memory, results_dir, run_key, write_table)

ABLATIONS = ("matched",)


def landmarks(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """(DAPI level-0 xy, H&E level-0 xy), paired row by row."""
    alignment = entry["alignment"]
    if alignment.get("kind") != "landmarks":
        raise ValueError(f"{entry['name']}: not a landmarks dataset")
    if "path" in alignment:
        if (alignment.get("fixed_space"), alignment.get("moving_space")) != ("dapi_level0_px", "he_level0_px"):
            raise ValueError(f"{entry['name']}: unexpected landmark spaces")
        with open(alignment["path"], newline="") as handle:
            rows = list(csv.DictReader(handle))
        dapi = np.array([[float(r["fixedX"]), float(r["fixedY"])] for r in rows])
        he = np.array([[float(r["alignmentX"]), float(r["alignmentY"])] for r in rows])
    else:
        sides = []
        for key in ("dapi_points", "he_points"):
            spec = alignment[key]
            sides.append(np.loadtxt(spec["path"], delimiter="\t", ndmin=2)[:, :2] * float(spec["scale_to_level0"]))
        dapi, he = sides
    if dapi.shape != he.shape or len(dapi) != alignment.get("n_points", len(dapi)):
        raise ValueError(f"{entry['name']}: landmark count mismatch")
    return dapi, he


def score(entry: dict, run_dir: Path, method: str, table: Path) -> dict:
    """Map the landmarks through one successful run; write its per-landmark table."""
    pm = PointMap(run_dir)
    if pm.kind not in METHOD_KINDS[method]:
        raise ValueError(f"{pm.kind} is not a {method} artifact")
    dapi, he = landmarks(entry)
    predicted, measured, reason = pm.apply(dapi)
    tre = np.linalg.norm(predicted - he, axis=1) * float(entry["he_mpp"])
    tre[~measured] = np.nan
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "dapi_x_level0", "dapi_y_level0", "he_x_level0", "he_y_level0",
                         "predicted_he_x_level0", "predicted_he_y_level0", "measured", "reason", "tre_um"])
        for i in range(len(dapi)):
            writer.writerow([i, *(f"{v:.6f}" for v in (*dapi[i], *he[i], *predicted[i])),
                             int(measured[i]), reason[i], f"{tre[i]:.6f}"])
    stats = distance_stats(tre)
    return {"kind": pm.kind, "n_points": len(dapi), **stats,
            "n_outside_he": int((reason == "target_outside_he").sum()),
            "n_unmeasured": int((~measured).sum()),
            "unmeasured_reasons": dict(Counter(reason[~measured])) or None,
            "complete": bool(measured.all()),
            "point_map": pm.describe()}


def best_by_dataset(rows: list[dict]) -> dict:
    """Lowest median TRE per dataset among complete runs, ablations excluded. This is a
    post hoc choice against the ground truth, not held-out performance."""
    best = {}
    for row in rows:
        if row.get("complete") and row["condition"] not in ABLATIONS:
            current = best.get(row["dataset"])
            if current is None or row["median_um"] < current["median_um"]:
                best[row["dataset"]] = {"condition": row["condition"], "median_um": row["median_um"],
                                        "n_points": row["n_points"]}
    return best


def evaluate(series: str, method: str, args) -> None:
    root = method_root(series, "keypoints", method)
    if not root.is_dir():
        print(f"[keypoints] {series} {method}: no runs under {root}; skipped", flush=True)
        return
    out = results_dir(series, "keypoints", method)
    rows = []
    for entry, condition, run_dir, record, selected in iter_runs(series, "keypoints", method,
                                                       args.datasets, args.conditions):
        row = {"dataset": entry["name"], "condition": condition, "outcome": record["outcome"],
               "reason": record.get("reason"), "method_sec": record.get("method_sec"),
               **peak_memory(run_dir, record)}
        if record["outcome"] != "success":
            rows.append(row)
            if selected:
                print(f"[keypoints] {series} {method} {condition:16s} {entry['name']}: "
                      f"{record['outcome']} ({record.get('reason')})", flush=True)
            continue
        table = out / "points" / entry["name"] / f"{condition}.csv"
        result_path = table.with_suffix(".json")
        result = cached_result(result_path, run_dir, args.rescore and selected)
        reused = result is not None
        if not reused and not selected:  # outside --datasets/--conditions: keep the table whole
            rows.append({**row, "eval_error": "not scored yet"})
            continue
        if not reused:
            try:
                result = {"scoring_code_sha256": SCORING_CODE_SHA256, "run_record_sha256": run_key(run_dir),
                          **score(entry, run_dir, method, table)}
                dump(result_path, result)
            except Exception as exc:
                result = {"eval_error": f"{type(exc).__name__}: {exc}"}
            gc.collect()  # a VALIS run's fields and splines are large; free them before the next
        row.update({k: v for k, v in result.items()
                    if k not in ("scoring_code_sha256", "run_record_sha256", "point_map", "unmeasured_reasons")})
        if result.get("unmeasured_reasons"):
            row["unmeasured_reasons"] = ";".join(f"{k}={v}" for k, v in result["unmeasured_reasons"].items())
        rows.append(row)
        if row.get("median_um") is None:
            line = row.get("eval_error") or "no landmark measured"
        else:
            line = f"median {row['median_um']:.2f} um ({row['n_scored']}/{row['n_points']}"
            line += (f", {row['n_outside_he']} outside H&E" if row["n_outside_he"] else "")
            line += (f", {row['n_unmeasured']} unmeasured" if row["n_unmeasured"] else "") + ")"
        if selected:
            print(f"[keypoints] {series} {method} {condition:16s} {entry['name']}: {line}"
                  + (" [cached]" if reused else ""), flush=True)
    write_table(out / "runs.csv", rows)
    dump(out / "summary.json", {"method": method, "series": series, "mode": "keypoints",
                                "scoring_code_sha256": SCORING_CODE_SHA256,
                                "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                                "metric": "TRE in H&E level-0 micrometres; every mapped landmark measured, "
                                          "including those mapped outside the H&E image",
                                "best_by_dataset": best_by_dataset(rows),
                                "runs": rows})
    print(f"[keypoints] {series}: {len(rows)} runs -> {out}", flush=True)


def main(method: str) -> int:
    args = parse_args("keypoints", method)
    for series in args.series:
        evaluate(series, method, args)
    return 0
