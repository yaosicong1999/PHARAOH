#!/usr/bin/env python3
"""Print the minitest's scores and how close they are to the benchmark's.

    python check.py        (4_evaluate.sh runs it after scoring)

For each dataset, the median, 95th percentile and largest error of this run (H&E um, from
results/minitest/<keypoints|matrix>/pharaoh/runs.csv, written by the benchmark's scoring), and:
  CLOSE   the median is within TOLERANCE_UM of the benchmark's median
  FAR     the median is further off
  FAIL    the run failed or was not scored

README.md lists the same metrics from four platforms. Processors, operating systems and system math
libraries change the last bits of floating-point results, so the transformation, and with it the
error, differs slightly between platforms; across four machines a dataset's median moved by at most
0.38 um (Supplementary Fig. 10), hence TOLERANCE_UM. Exits 1 on FAR or FAIL.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES = "minitest"
TOLERANCE_UM = 0.4
# dataset: (group, what is scored, the benchmark's median error in um)
DATASETS = {
    "human_crc_p1": ("keypoints", "landmark TRE", 0.51),
    "human_heart": ("matrix", "transcript vs 10x", 0.35),
}


def check(name: str, group: str, benchmark_um: float) -> dict:
    out = {"dataset": name, "benchmark_median_um": benchmark_um}
    record = HERE / "runs" / SERIES / group / "pharaoh" / "default" / name / "run_record.json"
    if not record.exists():
        return {**out, "verdict": "FAIL", "reason": "no run; run ./3_run.sh"}
    outcome = json.loads(record.read_text()).get("outcome")
    if outcome != "success":
        return {**out, "verdict": "FAIL", "reason": f"run outcome {outcome}"}
    table = HERE / "results" / SERIES / group / "pharaoh" / "runs.csv"
    row = next((r for r in csv.DictReader(table.open()) if r["dataset"] == name and r["condition"] == "default"),
               None) if table.exists() else None
    if row is None or not row.get("median_um"):
        return {**out, "verdict": "FAIL", "reason": "not scored; run ./4_evaluate.sh"}
    out.update({k: float(row[k]) for k in ("median_um", "p95_um", "max_um")})
    if abs(out["median_um"] - benchmark_um) <= TOLERANCE_UM:
        return {**out, "verdict": "CLOSE", "reason": f"median within {TOLERANCE_UM} um of the benchmark's"}
    return {**out, "verdict": "FAR", "reason": f"median more than {TOLERANCE_UM} um from the benchmark's"}


def main() -> int:
    print(f"\n{'dataset':13s} {'error (H&E um)':18s} {'median':>7s} {'P95':>6s} {'max':>6s} {'benchmark':>10s}  verdict")
    results = []
    for name, (group, metric, benchmark_um) in DATASETS.items():
        r = check(name, group, benchmark_um)
        results.append(r)
        cells = "".join(f" {r[k]:>6.2f}" if k in r else f" {'-':>6s}" for k in ("median_um", "p95_um", "max_um"))
        print(f"{name:13s} {metric:18s}{cells} {benchmark_um:>10.2f}  {r['verdict']}: {r['reason']}")
    bad = [r["dataset"] for r in results if r["verdict"] != "CLOSE"]
    print("\n[check] " + (f"{', '.join(bad)}: see README.md" if bad else
                          "compare with the four platforms in README.md"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
