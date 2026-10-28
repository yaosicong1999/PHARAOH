"""Finding runs to evaluate, and the shared command line of every eval script."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bench import manifest  # noqa: E402
from bench.config import CONFIGS, PACKAGES, environment, read_json  # noqa: E402

# A registration error above this is a failure: in the figures (failure band) and the summary
# (n_over_12um, the points of a run above it).
FAILURE_UM = 12.0

# The scoring code itself identifies the rules: a cached per-run result written by other
# code is recomputed. Any edit to a scoring module under eval/common/ therefore rescores every
# run; the figure style (figures.py) is not scoring code and is left out.
FIGURE_CODE = ("figures.py",)
SCORING_CODE_SHA256 = hashlib.sha256(b"".join(
    p.read_bytes() for p in sorted((ROOT / "eval" / "common").glob("*.py")) if p.name not in FIGURE_CODE)).hexdigest()

# The artifact kinds each method's wrapper can emit. An eval script refuses anything else,
# so a run cannot be scored by the wrong method's script by accident.
METHOD_KINDS = {"pharaoh": {"matrix"}, "voltron": {"matrix"}, "regwsi": {"regwsi_pull_field"},
                "valis_tifffile": {"matrix", "valis_native"}, "valis_bioformats": {"matrix", "valis_native"}}


def parse_args(mode: str, method: str):
    p = argparse.ArgumentParser(description=f"{mode} evaluation of {method} runs")
    p.add_argument("--series", nargs="+", required=True,
                   help="one or more output series, e.g. iter1 iter2 iter3 (runs/<series>/...)")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--conditions", nargs="+")
    p.add_argument("--rescore", action="store_true",
                   help="recompute runs already scored (by default a run is rescored only if its "
                        "run_record.json changed or the scoring code under eval/common/ did)")
    p.add_argument("--runs-root", type=Path,
                   help="the folder holding <series>/<group>/<method>/ (default: runs_root of "
                        "configs/environment.json)")
    args = p.parse_args()
    if args.runs_root:
        if not args.runs_root.is_dir():
            raise SystemExit(f"--runs-root {args.runs_root} is not a folder")
        global RUNS_ROOT
        RUNS_ROOT = args.runs_root
    expected = Path(environment()["eval_python"][PACKAGES[method]]).expanduser()
    if Path(sys.executable).resolve() != expected.resolve():
        print(f"[warn] {method} eval is configured for {expected}; running under {sys.executable}",
              flush=True)
    return args


RUNS_ROOT = None  # set by --runs-root


def method_root(series: str, group: str, method: str) -> Path:
    return Path(RUNS_ROOT or environment()["runs_root"]) / series / group / method


def condition_order(method: str) -> list[str]:
    """The method's conditions in config order: main config first, then the ablations."""
    order = []
    for path in [CONFIGS / "methods" / f"{method}.json", *sorted(CONFIGS.glob(f"ablation_*/{method}.json"))]:
        if path.exists():
            order += [c for c in read_json(path)["conditions"] if c not in order]
    return order


def iter_runs(series: str, group: str, method: str, datasets=None, conditions=None):
    """Every (dataset, condition) with a run record, successful or not, as
    (entry, condition, run_dir, record, selected); dataset-major, the conditions in config
    order (any condition no config names comes last, by name). All runs are yielded so the
    method's tables stay whole; `selected` says whether --datasets/--conditions chose it."""
    root = method_root(series, group, method)
    if not root.exists():
        raise SystemExit(f"no runs under {root}")
    found = {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")}
    order = condition_order(method)
    names = [c for c in order if c in found] + sorted(found - set(order))
    manifest.load(group, datasets)  # rejects an unknown dataset name
    for entry in manifest.load(group):
        for condition in names:
            run_dir = root / condition / entry["name"]
            record_path = run_dir / "run_record.json"
            if record_path.exists():
                selected = ((not datasets or entry["name"] in datasets)
                            and (not conditions or condition in conditions))
                yield entry, condition, run_dir, read_json(record_path), selected


def peak_memory(run_dir: Path, record: dict) -> dict:
    """A run's peak memory in MiB: the runner's process-tree RSS, plus the GPU memory the method's
    wrapper recorded (run_metadata.json peak_gpu_mib; RegWSI on MPS). Unified-memory GPU buffers
    barely reach RSS, so the two are added; the sum of the two peaks is an upper bound when they
    do not coincide. A CPU-only run has peak_gpu_mib 0."""
    rss = record.get("peak_rss_mib")
    metadata = Path(run_dir) / "run_metadata.json"
    gpu = read_json(metadata).get("peak_gpu_mib") if metadata.exists() else None
    gpu = float(gpu) if gpu is not None else 0.0
    return {"peak_rss_mib": rss, "peak_gpu_mib": gpu,
            "peak_memory_mib": float(rss) + gpu if rss is not None else None}


def run_key(run_dir: Path) -> str:
    """Identifies the scored run: a rerun rewrites run_record.json and changes this."""
    return hashlib.sha256((Path(run_dir) / "run_record.json").read_bytes()).hexdigest()


def cached_result(path: Path, run_dir: Path, rescore: bool):
    """A per-run result written earlier for this very run under the current rules, else None."""
    if rescore or not path.exists():
        return None
    result = read_json(path)
    if (result.get("scoring_code_sha256") != SCORING_CODE_SHA256
            or result.get("run_record_sha256") != run_key(run_dir)):
        return None
    return result


def results_dir(series: str, mode: str, method: str) -> Path:
    path = Path(environment()["results_root"]) / series / mode / method
    path.mkdir(parents=True, exist_ok=True)
    return path


def distance_stats(distances_um: np.ndarray) -> dict:
    d = np.asarray(distances_um, dtype=float)
    d = d[np.isfinite(d)]
    if not len(d):
        return {"n_scored": 0}
    q = np.quantile(d, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {"n_scored": int(len(d)), "mean_um": float(d.mean()), "rmse_um": float(np.sqrt(np.mean(d ** 2))),
            "p25_um": float(q[0]), "median_um": float(q[1]), "p75_um": float(q[2]),
            "p90_um": float(q[3]), "p95_um": float(q[4]), "p99_um": float(q[5]), "max_um": float(d.max())}


def write_table(path: Path, rows: list[dict]) -> None:
    import csv
    columns = []
    for row in rows:
        columns += [k for k in row if k not in columns]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
