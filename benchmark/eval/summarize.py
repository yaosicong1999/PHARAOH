#!/usr/bin/env python3
"""One table per mode across methods and series, from the per-method eval results.

    ~/miniconda3/envs/PHARAOH/bin/python eval/summarize.py --mode keypoints --series iter1 iter2 iter3
    ~/miniconda3/envs/PHARAOH/bin/python eval/summarize.py --mode keypoints --series iter1 --time-series iter1 iter2 iter3

Reads results/<series>/<mode>/<method>/runs.csv (written by eval/<mode>/<method>.py; run
those first) and writes results/summary/<series>/<mode>/ (several series: <s1>-<s2>-..., e.g. iter1-iter2-iter3)
    runs.csv          every run of every method and series, one row each
    conditions.csv    one row per (method, dataset, condition): its runs over the series
    best.csv          per (method, dataset), the condition with the lowest median error, with
                      n_over_12um / n_scored: how many of its points are above the failure line
    summary.json      the inputs, and which methods or series were missing

A condition is eligible for best.csv only if it succeeded and was fully scored in every
series given (keypoints: every landmark measured), and is not the `matched` ablation. The
choice is post hoc against the reference, not held-out performance.

--time-series averages the time over other series than the accuracy: errors and peak memory
come from --series (e.g. iter1, the scored reference; iter2/iter3 keep only logs and timing and
are not scored), the time from the run records of every --time-series run (run_record.json
under the runs root; no evaluation needed). method_sec_mean / method_sec_sd / wall_sec_mean are
then over those series' successful runs, with method_sec_<series> per series, n_time_success
and time_complete (succeeded in every time series). Written to
results/summary/<series>__time_<s1>-<s2>-.../<mode>/, e.g. iter1__time_iter1-iter2-iter3; the
visualization scripts take the same two options and read it from there.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path[0] = str(Path(__file__).resolve().parents[1])
from bench.config import EVAL_MODES, METHODS, environment  # noqa: E402
from eval.common.runs import FAILURE_UM, SCORING_CODE_SHA256, condition_order, dump, write_table  # noqa: E402

OVER = f"n_over_{FAILURE_UM:g}um"            # points of a run above the failure threshold
OVER_FRACTION = f"fraction_over_{FAILURE_UM:g}um"

ABLATIONS = ("matched",)
ERROR_COLUMNS = ("median_um", "mean_um", "p90_um", "max_um")


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fully_scored(row: dict, mode: str) -> bool:
    if row["outcome"] != "success" or number(row.get("median_um")) is None:
        return False
    if mode == "keypoints":
        return row.get("complete") == "True"
    return True


def points_over(results: Path, row: dict, mode: str):
    """(points above FAILURE_UM, points scored) of one successful run, from its per-run result:
    the keypoint table, or the transcript histogram (whose edges include FAILURE_UM)."""
    base = results / row["series"] / mode / row["method"]
    if mode == "keypoints":
        table = base / "points" / row["dataset"] / f"{row['condition']}.csv"
        if not table.exists():
            return None, None
        values = [float(r["tre_um"]) for r in csv.DictReader(table.open()) if r["measured"] == "1"]
        return sum(v > FAILURE_UM for v in values), len(values)
    result = base / "runs" / row["dataset"] / f"{row['condition']}.json"
    if not result.exists():
        return None, None
    histogram = json.loads(result.read_text())["histogram"]
    edges = [np.inf if e is None else e for e in histogram["edges_um"]]
    over = sum(c for left, c in zip(edges[:-1], histogram["counts"]) if left >= FAILURE_UM - 1e-9)
    return int(over), int(sum(histogram["counts"]))


GPU_METHODS = ("regwsi",)   # the only method whose wrapper allocates GPU memory (MPS)


def memory_mib(method: str, row: dict):
    """Peak memory (RSS + GPU, eval/common/runs.py peak_memory) of a runs.csv row. A table written
    before that column existed is read as RSS, which is the same for a CPU-only method."""
    value = number(row.get("peak_memory_mib"))
    if value is not None or "peak_memory_mib" in row and row["peak_memory_mib"] not in ("", None):
        return value
    if method in GPU_METHODS and number(row.get("peak_rss_mib")) is not None:
        raise SystemExit(f"{method}: the evaluation has no peak_memory_mib (RSS + GPU) yet: re-run "
                         f"eval/eval_{method}.sh --mode keypoints|matrix")
    return number(row.get("peak_rss_mib"))


def summary_name(series: list[str], time_series: list[str] | None = None) -> str:
    """The folder of a summary under results/summary/ (and of its figures under results/visualization/)."""
    name = "-".join(series)
    return f"{name}__time_{'-'.join(time_series)}" if time_series else name


def time_average(runs_root: Path, series: list[str], mode: str, methods: list[str]) -> dict:
    """{(method, dataset, condition): time statistics} over the given series' run records.
    Only successful runs count towards the means; a series without the run, or where it
    failed, shows as missing in method_sec_<series> and n_time_success."""
    found = {}
    for s in series:
        for method in methods:
            root = runs_root / s / mode / method
            if not root.is_dir():
                continue
            for record_path in root.glob("*/*/run_record.json"):
                condition, dataset = record_path.parent.parent.name, record_path.parent.name
                if condition.startswith("_"):
                    continue
                record = json.loads(record_path.read_text())
                found.setdefault((method, dataset, condition), {})[s] = record
    stats = {}
    for key, records in found.items():
        ok = {s: r for s, r in records.items() if r.get("outcome") == "success" and r.get("method_sec") is not None}
        seconds = [float(r["method_sec"]) for r in ok.values()]
        walls = [float(r["wall_seconds"]) for r in ok.values() if r.get("wall_seconds") is not None]
        row = {"time_series": "-".join(series), "n_time_success": len(ok), "time_complete": len(ok) == len(series),
               "method_sec_mean": float(np.mean(seconds)) if seconds else None,
               "method_sec_sd": float(np.std(seconds, ddof=1)) if len(seconds) > 1 else None,
               "method_sec_cv": (float(np.std(seconds, ddof=1) / np.mean(seconds)) if len(seconds) > 1 else None),
               "wall_sec_mean": float(np.mean(walls)) if walls else None,
               "wall_sec_sd": float(np.std(walls, ddof=1)) if len(walls) > 1 else None}
        for s in series:
            row[f"method_sec_{s}"] = float(ok[s]["method_sec"]) if s in ok else None
        stats[key] = row
    return stats


def spread(values):
    """mean, min, max of the non-missing values (None when there are none)."""
    values = [v for v in values if v is not None]
    if not values:
        return None, None, None
    return float(np.mean(values)), float(min(values)), float(max(values))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", required=True, choices=EVAL_MODES)
    p.add_argument("--series", nargs="+", required=True, help="e.g. iter1 iter2 iter3")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--time-series", nargs="+",
                   help="average the time over these series instead (e.g. iter1 iter2 iter3); errors and "
                        "memory stay from --series")
    p.add_argument("--runs-root", type=Path,
                   help="folder holding <series>/ for --time-series (default: runs_root of configs/environment.json)")
    args = p.parse_args()

    results = Path(environment()["results_root"])
    runs, missing = [], []
    for method in args.methods:
        for series in args.series:
            path = results / series / args.mode / method / "runs.csv"
            if not path.exists():
                missing.append(f"{series}/{method}")
                continue
            with path.open(newline="") as handle:
                runs += [{"series": series, "method": method, **row} for row in csv.DictReader(handle)]
    if not runs:
        raise SystemExit(f"no results under {results}/<series>/{args.mode}/ for {args.series}")

    for row in runs:
        if row["outcome"] == "success" and number(row.get("median_um")) is not None:
            over, scored = points_over(results, row, args.mode)
            row[OVER], row[OVER_FRACTION] = over, (over / scored if scored else None)

    out = results / "summary" / summary_name(args.series, args.time_series) / args.mode
    out.mkdir(parents=True, exist_ok=True)
    write_table(out / "runs.csv", runs)

    groups = defaultdict(list)
    for row in runs:
        groups[(row["method"], row["dataset"], row["condition"])].append(row)
    datasets = list(dict.fromkeys(row["dataset"] for row in runs))
    order = {m: condition_order(m) for m in args.methods}

    def sort_key(key):
        method, dataset, condition = key
        conditions = order[method]
        return (args.methods.index(method), datasets.index(dataset),
                conditions.index(condition) if condition in conditions else len(conditions), condition)

    conditions = []
    for key in sorted(groups, key=sort_key):
        method, dataset, condition = key
        rows = {row["series"]: row for row in groups[key]}
        present = [rows[s] for s in args.series if s in rows]
        row = {"method": method, "dataset": dataset, "condition": condition,
               "n_series": len(present),
               "n_success": sum(r["outcome"] == "success" for r in present),
               "fully_scored_in_all": len(present) == len(args.series)
                                      and all(fully_scored(r, args.mode) for r in present),
               "outcomes": ";".join(f"{r['series']}={r['outcome']}" + (f"({r['reason']})" if r.get("reason") else "")
                                    for r in present if r["outcome"] != "success") or None}
        for column in ERROR_COLUMNS:
            mean, low, high = spread(number(r.get(column)) for r in present)
            row[f"{column}"] = mean
            if column == "median_um":
                row["median_um_min"], row["median_um_max"] = low, high
        seconds = [number(r.get("method_sec")) for r in present if r["outcome"] == "success"]
        seconds = [s for s in seconds if s is not None]
        row["method_sec_mean"] = float(np.mean(seconds)) if seconds else None
        row["method_sec_sd"] = float(np.std(seconds, ddof=1)) if len(seconds) > 1 else None
        for column, value_of in (("peak_rss_mib", lambda r: number(r.get("peak_rss_mib"))),
                                 ("peak_gpu_mib", lambda r: number(r.get("peak_gpu_mib")) or 0.0),
                                 ("peak_memory_mib", lambda r: memory_mib(method, r))):   # memory = RSS + GPU
            values = [value_of(r) for r in present if number(r.get("peak_rss_mib")) is not None]
            values = [v for v in values if v is not None]
            row[f"{column}_max"] = max(values) if values else None
        overs = [r.get(OVER) for r in present if r.get(OVER) is not None]
        fractions = [r.get(OVER_FRACTION) for r in present if r.get(OVER_FRACTION) is not None]
        row[OVER] = max(overs) if overs else None                       # worst series
        row[OVER_FRACTION] = max(fractions) if fractions else None
        scored = [number(r.get("n_scored")) for r in present if number(r.get("n_scored")) is not None]
        row["n_scored"] = int(max(scored)) if scored else None
        for column in ("n_outside_he", "n_unmeasured", "n_invalid"):
            values = [number(r.get(column)) for r in present if number(r.get(column)) is not None]
            if values:
                row[f"{column}_max"] = int(max(values))
        conditions.append(row)
    times = {}
    if args.time_series:
        times = time_average(args.runs_root or Path(environment()["runs_root"]), args.time_series, args.mode, args.methods)
        for row in conditions:
            timing = times.get((row["method"], row["dataset"], row["condition"]))
            row.update(timing or {"time_series": "-".join(args.time_series), "n_time_success": 0,
                                  "time_complete": False, "method_sec_mean": None, "method_sec_sd": None})
    write_table(out / "conditions.csv", conditions)

    best = {}
    for row in conditions:
        if not row["fully_scored_in_all"] or row["condition"] in ABLATIONS:
            continue
        key = (row["method"], row["dataset"])
        if key not in best or row["median_um"] < best[key]["median_um"]:
            best[key] = row
    best_rows = []
    for method in args.methods:
        for dataset in datasets:
            row = best.get((method, dataset))
            best_rows.append({"method": method, "dataset": dataset,
                              **({k: row[k] for k in ("condition", "median_um", "median_um_min", "median_um_max",
                                                      "mean_um", "p90_um", OVER, "n_scored", OVER_FRACTION,
                                                      "method_sec_mean", "method_sec_sd", "peak_memory_mib_max",
                                                      "peak_rss_mib_max", "peak_gpu_mib_max")
                                    if k in row}
                                 if row else {"condition": None})})
    write_table(out / "best.csv", best_rows)

    dump(out / "summary.json", {"mode": args.mode, "series": args.series, "time_series": args.time_series,
                                "methods": args.methods,
                                "scoring_code_sha256": SCORING_CODE_SHA256,
                                "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                                "missing": missing, "n_runs": len(runs), "n_conditions": len(conditions),
                                "best": best_rows})
    width = max(len(d) for d in datasets)
    print(f"[summary] {args.mode} over {', '.join(args.series)}: {len(runs)} runs -> {out}")
    print(f"[summary] best = lowest median; * = that run has points above {FAILURE_UM:g} um ({OVER} in best.csv)")
    if args.time_series:
        print(f"[summary] time = mean over {', '.join(args.time_series)} (run records); errors and memory from "
              f"{', '.join(args.series)}")
        for method in args.methods:
            rows = [r for r in conditions if r["method"] == method and r.get("n_time_success")]
            partial = [r for r in rows if not r["time_complete"]]
            if partial:   # timed somewhere, but not in every time series: its mean rests on fewer runs
                gaps = ", ".join(f"{s} {sum(r.get(f'method_sec_{s}') is None for r in partial)}" for s in args.time_series)
                print(f"[summary] {method}: {len(partial)}/{len(rows)} timed conditions lack a successful run in some "
                      f"time series (missing: {gaps}); n_time_success says how many runs each mean has")
    if missing:
        print(f"[summary] missing (score them first: run_eval.sh, or eval/eval_<method>.sh --mode {args.mode}): {', '.join(missing)}")
    print(f"{'dataset':{width}s}  " + "  ".join(f"{m:>22s}" for m in args.methods))
    for dataset in datasets:
        cells = []
        for method in args.methods:
            row = best.get((method, dataset))
            star = "*" if row and row.get(OVER) else " "
            cells.append(f"{row['median_um']:8.2f}{star} {row['condition']:>12s}" if row else f"{'-':>22s}")
        print(f"{dataset:{width}s}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
