#!/usr/bin/env python3
"""Make the VALIS input re-encodes of one series before its VALIS batches run.

    <python.valis> prepare_valis_inputs.py --series iter2 --manifest manifests/keypoints.json
    <python.valis> prepare_valis_inputs.py --series iter1 --groups matrix --datasets human_crc

run_benchmark.sh calls it (prepare_valis_inputs line) right before the VALIS batches.
Uses the wrapper's own functions (methods/valis/wrapper.py), as the VALIS methods read
their inputs: every DAPI is re-encoded (decoded by tifffile; valis_tifffile reads it), and
an H&E only if VALIS's reader does not return the same pixels as tifffile (both methods).

The conversion is redone once per series: a file without a conversion for --series is
decoded and re-encoded again into <cache_root>/valis_inputs/ (overwriting the cached file)
and that time is recorded for the series beside it; every VALIS run of the series charges
it. Files already converted for the series are skipped, so a repeated call is quick.

Writes <cache_root>/valis_inputs/prepared.json: per input file, the datasets that use it,
direct or re-encoded (and why), the cached path, its size and the conversion per series.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[0] = str(ROOT)
from bench import manifest  # noqa: E402
from bench.config import environment, write_json  # noqa: E402

spec = importlib.util.spec_from_file_location("valis_wrapper", ROOT / "methods/valis/wrapper.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series", required=True, help="the series whose conversion this is, e.g. iter2")
    p.add_argument("--manifest", nargs="+", type=Path, help="manifest file(s); default: --groups")
    p.add_argument("--groups", nargs="+", default=["keypoints", "matrix"])
    p.add_argument("--datasets", nargs="+")
    args = p.parse_args()
    entries = ([(g, e) for path in args.manifest for g, ds in [manifest.load_file(path)] for e in ds]
               if args.manifest else [(g, e) for g in args.groups for e in manifest.load(g)])

    cache = Path(environment()["cache_root"]) / "valis_inputs"
    cache.mkdir(parents=True, exist_ok=True)
    # One entry per distinct file, in manifest order; DAPI first, as a run reads them.
    inputs = {}
    for group, entry in entries:
        if args.datasets and entry["name"] not in args.datasets:
            continue
        for modality, name in (("dapi", "0_dapi.ome.tiff"), ("he", "1_he.ome.tiff")):
            path = str(Path(entry[modality]).resolve())
            item = inputs.setdefault(path, {"modality": modality, "name": name, "used_by": []})
            item["used_by"].append(f"{group}/{entry['name']}")
    print(f"[prepare] series {args.series}: {len(inputs)} input files -> {cache}", flush=True)

    report_path = cache / "prepared.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"files": {}}
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory() as scratch:
            for i, (path, item) in enumerate(inputs.items(), 1):
                label = f"[{i}/{len(inputs)}] {item['modality']} {item['used_by'][0]}"
                t0 = time.monotonic()
                try:
                    # As valis_tifffile reads them: the DAPI always re-encoded by tifffile,
                    # the H&E only if VALIS's reader does not read it correctly.
                    result = wrapper.native_input(Path(path), item["name"], Path(scratch), cache,
                                                  probe=item["modality"] == "he", series=args.series)
                except Exception as exc:  # one bad file must not stop the rest
                    report["files"][path] = {**item, "error": f"{type(exc).__name__}: {exc}"}
                    print(f"{label}: FAILED {type(exc).__name__}: {exc}", flush=True)
                    write_json(report_path, report)
                    continue
                state = ("direct" if result["direct"] else
                         f"converted for {args.series} already" if result["cached"] else "re-encoded")
                size = None if result["direct"] else Path(result["path"]).stat().st_size
                report["files"][path] = {
                    **item, "direct": result["direct"], "reader_error": result.get("reader_error"),
                    "cached_path": None if result["direct"] else result["path"],
                    "cached_bytes": size,
                    "conversions": None if result["direct"] else wrapper.cache_conversions(
                        Path(result["path"] + ".json")),
                    "levels": result["available_levels"]}
                print(f"{label}: {state}"
                      + ("" if result["direct"] else
                         f", {size / 2**30:.2f} GiB, conversion {result['conversion_sec']:.0f}s")
                      + f" ({time.monotonic() - t0:.0f}s)", flush=True)
                report["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
                write_json(report_path, report)
    finally:
        from valis import registration
        registration.kill_jvm()

    files = report["files"].values()
    encoded = [f for f in files if f.get("direct") is False]
    print(f"[prepare] done in {(time.monotonic() - started) / 60:.1f} min: "
          f"{sum(1 for f in files if f.get('direct'))} direct, {len(encoded)} re-encoded "
          f"({sum(f['cached_bytes'] for f in encoded) / 2**30:.1f} GiB), "
          f"{sum(1 for f in files if 'error' in f)} failed -> {report_path}", flush=True)
    return 1 if any("error" in f for f in files) else 0


if __name__ == "__main__":
    raise SystemExit(main())
