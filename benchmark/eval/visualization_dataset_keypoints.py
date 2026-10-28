#!/usr/bin/env python3
"""Per-dataset keypoint (or matrix) figures: accuracy per method, across resolutions, and against cost.

    <eval_python.pharaoh> eval/visualization_dataset_keypoints.py --series iter1 [--group matrix] [--datasets ...] [--time method|wall]

--group matrix draws the same figures for the matrix datasets, from results/<series>/matrix/<method>/
(runs.csv, runs/<dataset>/<condition>.json): the y axis is each run's median distance of every
transcript to 10x's alignment (agreement with 10x, not accuracy); the box plot is drawn from the
stored percentiles (box p25-p75, whiskers p5-p95, no points; p5 from the 0.1 µm histogram); * = a
transcript above 12 µm; files are named *_distance_* instead of *_tre_*; output under
results/visualization/<series>/matrix/<dataset>/. The rest of this note describes the keypoint group.

Reads results/<series>/keypoints/<method>/ (runs.csv, points/<dataset>/<condition>.csv; written by
eval/eval_<method>.sh --mode keypoints) and, for --time wall, each run's run_record.json. For every
keypoint dataset it writes two views:

    results/visualization/<series>/keypoints/<dataset>/                       PHARAOH, VoltRon, RegWSI, VALIS (tifffile, labelled "VALIS")
    results/visualization/<series>/keypoints/ablation_bioformats/<dataset>/   the same plus VALIS (Bio-Formats)

each holding, as 300-DPI PNG and editable PDF:

    <dataset>_tre_boxplot                      every keypoint TRE of each method's best condition
    <dataset>_resolution_tre                   median TRE across the resolution caps, one line per preset family
    <dataset>_runtime_tre                      median TRE against time
    <dataset>_memory_tre                       median TRE against peak memory (RSS + GPU)
    <dataset>_runtime_tre_with_size_memory     median TRE against time, point area = peak memory (2:1)

plus <dataset>_conditions.csv (every plotted run), manifest.json and README.md.

Time (--time): method_sec (default; the method's own time, its input preparation included, as
the benchmark charges it) or wall (the run's wall-clock time). With --time-series iter1 iter2
iter3 the time is the mean over those series (± SD as horizontal bars), read from the summary
eval/summarize.py --series <series> --time-series ... writes; accuracy and memory stay from
--series. Those figures go to results/visualization/<series>__time_<s1>-<s2>-.../keypoints/. Every successful, scored,
non-matched run is plotted; the box plot uses each method's lowest-median, fully scored one
(post hoc against the ground truth). A preset family is one method's line across resolution
caps: RegWSI initial / nonrigid / hires, VALIS rigid / nonrigid / micro. Shared style with
eval/visualization_combine_keypoints.py (eval/common/figures.py): failure band above 12 µm with
method-coloured crosses, medians in the method's colour, * = a keypoint above 12 µm.

Adapted from revision/benchmark/code/plot_keypoint_dataset_metrics.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[0] = str(ROOT)
from bench import manifest  # noqa: E402
from bench.config import environment, read_json  # noqa: E402
from eval.summarize import memory_mib, summary_name  # noqa: E402
from eval.common.figures import (ABLATIONS, COLORS, FAIL_COLOR, FAIL_MARK_Y, FAIL_MARKER, LABELS, MAIN_METHODS,  # noqa: E402
                                 METHODS, OUTLINE, SECONDARY, Y_CAP, Y_TOP, accuracy_bands, box_top, fail_text_y,
                                 alphabetical, condition_label, flip_note, label, median_label, method_label, plt, save,
                                 save_legend, use_labels)

TIERS = ("1k", "2k", "4k", "8k", "16k")
# what differs between the keypoint and the matrix figures (set in main from --group)
GROUPS = {"keypoints": {"metric": "tre", "median_axis": "Median TRE (µm)",
                        "box_axis": "TRE vs 10x annotated keypoint (µm)", "what": "Accuracy",
                        "points": "keypoint"},
          "matrix": {"metric": "distance", "median_axis": "Median distance to 10x alignment (µm)",
                     "box_axis": "Transcript distance to 10x alignment (µm)", "what": "Agreement with 10x",
                     "points": "transcript"}}
G = dict(GROUPS["keypoints"], group="keypoints")
PRESETS = ("initial", "rigid", "nonrigid", "hires", "micro")   # legend order within a method: RegWSI initial, nonrigid, hires; VALIS rigid, nonrigid, micro
# Within a method, the preset family is told apart by line pattern and marker.
# RegWSI and VALIS share the scheme, step by step: initial / rigid = solid line, filled circle;
# nonrigid = dotted line, hollow circle; hires / micro = dashed line, filled square.
DETAIL_STYLE = {"": ("-", "o", True),
                "initial": ("-", "o", True), "rigid": ("-", "o", True),
                "nonrigid": ((0, (1, 1.5)), "o", False),
                "hires": ("--", "s", True), "micro": ("--", "s", True)}


# ---- data -------------------------------------------------------------------------------

def tier_of(condition: str):
    tier = condition.rsplit("_", 1)[-1]
    return tier if tier in TIERS else None


def family_of(method: str, condition: str) -> str:
    if method in ("pharaoh", "voltron"):
        return method
    return f"{method}:{condition.rsplit('_', 1)[0]}"


def family_label(family: str) -> str:
    method, _, detail = family.partition(":")
    return method_label(method, detail)   # VALIS-rigid, RegWSI-hires, ...


def load_runs(results: Path, runs_root: Path, series: str, dataset: str, methods: list[str], time_key: str,
              timing: dict | None = None) -> list[dict]:
    """Every successful, scored, non-matched run of the dataset, with its keypoint TREs. With
    `timing` (the averaged summary's rows by (method, dataset, condition)) the time is that mean.
    A run that returned no transformation (a guard stop, kill or error) is included as failed:
    drawn as a cross in the failure band, at the time and memory it had used when it stopped."""
    rows = []
    for method in methods:
        table = results / method / "runs.csv"
        if not table.exists():
            print(f"[visualization] {method}: no evaluation at {table}; left out", flush=True)
            continue
        for row in csv.DictReader(table.open()):
            if row["dataset"] != dataset or row["condition"] in ABLATIONS:
                continue
            if row["outcome"] != "success":
                rows.append(failed_run(runs_root / series / G["group"] / method / row["condition"] / dataset,
                                       method, row))
                continue
            if not row.get("median_um"):
                continue
            if G["group"] == "matrix":   # percentiles and a histogram only: no per-point values
                values, box, n_over, n_points = matrix_box(results / method / "runs" / dataset / f"{row['condition']}.json")
            else:
                points = results / method / "points" / dataset / f"{row['condition']}.csv"
                values = np.array([float(r["tre_um"]) for r in csv.DictReader(points.open()) if r["measured"] == "1"])
                box, n_over, n_points = None, int((values > Y_CAP).sum()), len(values)
            if time_key == "wall":
                record = read_json(runs_root / series / G["group"] / method / row["condition"] / dataset / "run_record.json")
                seconds = record.get("wall_seconds")
            else:
                seconds = float(row["method_sec"]) if row.get("method_sec") else None
            seconds_sd, n_time = None, 1
            if timing is not None:
                averaged = timing.get((method, dataset, row["condition"]), {})
                kind = "method" if time_key == "method" else "wall"
                number = lambda key: float(averaged[key]) if averaged.get(key) not in (None, "") else None  # noqa: E731
                seconds, seconds_sd = number(f"{kind}_sec_mean"), number(f"{kind}_sec_sd")
                n_time = int(averaged.get("n_time_success") or 0)
            rows.append({"method": method, "condition": row["condition"], "family": family_of(method, row["condition"]),
                         "tier": tier_of(row["condition"]), "median_um": float(row["median_um"]), "values": values,
                         "box": box, "n_over": n_over, "n_points": n_points,
                         "complete": row.get("complete") == "True" or G["group"] == "matrix",
                         "seconds": seconds, "seconds_sd": seconds_sd,
                         "n_time": n_time, "failed": False,
                         "peak_gib": peak_gib(method, row)})
    return rows


def matrix_box(path: Path):
    """(no values, box statistics, transcripts above Y_CAP, transcripts scored) of a matrix run."""
    from eval.visualization_combine_matrix import percentile_from_histogram
    run = read_json(path)
    histogram = run["histogram"]
    edges = [np.inf if e is None else e for e in histogram["edges_um"]]
    over = int(sum(c for left, c in zip(edges[:-1], histogram["counts"]) if left >= Y_CAP - 1e-9))
    box = {"med": run["median_um"], "q1": run["p25_um"], "q3": run["p75_um"], "whislo": percentile_from_histogram(histogram, 5),
           "whishi": run["p95_um"], "fliers": []}
    return np.array([]), box, over, int(run["n_scored"])


def star_label(r: dict) -> str:
    """The median, with * when any point (keypoint or transcript) of the run is above the failure line."""
    return f"{r['median_um']:.2f}" + ("*" if r.get("n_over") else "")


def failed_run(run_dir: Path, method: str, row: dict) -> dict:
    """A run that returned no transformation: plotted in the failure band at its resolution cap,
    at the wall time until it stopped, and at its peak memory until then. A RegWSI run stopped by
    its MPS budget exits before writing its GPU peak; that peak is read from the stop message."""
    record_path = run_dir / "run_record.json"
    record = read_json(record_path) if record_path.exists() else {}
    memory = peak_gib(method, row)
    log = run_dir / "run.log"
    if row.get("reason") == "mps_budget" and log.exists():
        found = re.search(r"MPS_BUDGET_EXCEEDED: ([0-9.]+) GiB", log.read_text(errors="replace"))
        if found:
            memory = (memory or 0.0) + float(found.group(1))
    return {"method": method, "condition": row["condition"], "family": family_of(method, row["condition"]),
            "tier": tier_of(row["condition"]), "median_um": float("inf"), "values": np.array([]),
            "box": None, "n_over": 0, "n_points": 0,
            "complete": False, "seconds": record.get("wall_seconds"), "seconds_sd": None, "n_time": 1,
            "failed": True, "peak_gib": memory}


def peak_gib(method: str, row: dict):
    """Peak memory of a run (process-tree RSS + GPU; eval/summarize.py memory_mib) in GiB."""
    mib = memory_mib(method, row)
    return mib / 1024 if mib is not None else None


def family_style(family: str) -> dict:
    method, _, detail = family.partition(":")
    linestyle, marker, filled = DETAIL_STYLE.get(detail, ("-", "o", True))
    thin = method in SECONDARY   # VALIS (Bio-Formats): thinner line, besides its own colour
    return {"color": COLORS[method], "linestyle": linestyle, "marker": marker,
            "markerfacecolor": COLORS[method] if filled else "white",
            "markeredgecolor": "white" if filled else COLORS[method],
            "markeredgewidth": 0.9 if filled else 1.7, "linewidth": 1.4 if thin else 2.0, "markersize": 6.5}


def legend_columns(n: int) -> int:
    """Two legend rows, filled column by column: PHARAOH / VoltRon in the first column, then the
    RegWSI and VALIS presets in order (initial, nonrigid, hires, rigid, nonrigid, micro)."""
    return max(1, -(-n // 2))


def families(rows: list[dict], methods: list[str]) -> list[str]:
    names = {r["family"] for r in rows if r["method"] != "pharaoh" and r["tier"]}
    return sorted(names, key=lambda f: (methods.index(f.split(":")[0]), PRESETS.index(f.split(":")[1])
                                        if ":" in f and f.split(":")[1] in PRESETS else 0))


def shown_y(median: float) -> float:
    return FAIL_MARK_Y if median > Y_CAP else median


def heading(fig, title: str) -> None:
    fig.suptitle(title, x=0.5, y=0.975, fontsize=15)


PAGE = (12, 6)   # the line figures: a fixed 2:1 page (12 x 6 in)


def wide_figure():
    fig, ax = plt.subplots(figsize=PAGE)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.9, bottom=0.11)
    return fig, ax


def fit_page(fig, ax) -> None:
    """No white margin on the left, bottom and right: the axes grow until their labels and tick
    labels touch those page edges (the top keeps the title's room)."""
    for _ in range(3):   # the tick labels move a little with the axes; settles in two passes
        fig.canvas.draw()
        box = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        pos = ax.get_position()
        fig.subplots_adjust(left=pos.x0 - box.x0, bottom=pos.y0 - box.y0, right=1 - (box.x1 - pos.x1),
                            top=fig.subplotpars.top)


def save_page(fig, base: Path) -> list[Path]:
    """No tight bounding box, so the page keeps exactly its 2:1 size; no white margin on the
    left, bottom and right (fit_page)."""
    fit_page(fig, fig.axes[0])
    base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".png", ".pdf"):
        path = base.with_suffix(suffix)
        fig.savefig(path, facecolor="white", **({"dpi": 300} if suffix == ".png" else {}))
        written.append(path)
    plt.close(fig)
    return written


# ---- figures ----------------------------------------------------------------------------

def plot_boxplot(rows, methods, title, out, stem) -> list[Path]:
    best = {}
    for r in rows:
        if r["complete"] and (r["method"] not in best or r["median_um"] < best[r["method"]]["median_um"]):
            best[r["method"]] = r
    shown = [m for m in methods if m in best]
    fig, ax = plt.subplots(figsize=(max(5.5, 1.45 * len(shown) + 1.6), 5.9))
    accuracy_bands(ax)
    clip = plt.Rectangle((0, 0), 1, Y_CAP, transform=ax.get_yaxis_transform())
    top_of = (lambda r: r["box"]["whishi"]) if G["group"] == "matrix" else (lambda r: box_top(r["values"]))
    in_band = [m for m in shown if best[m]["median_um"] > Y_CAP or top_of(best[m]) >= Y_CAP]
    for i, method in enumerate(shown, start=1):
        pick, color = best[method], COLORS[method]
        dash = SECONDARY.get(method, {}).get("linestyle", "solid")
        style = dict(medianprops={"color": color, "linewidth": 2.2},
                     boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.8, "linestyle": dash},
                     whiskerprops={"color": color, "linewidth": 1.25, "linestyle": dash},
                     capprops={"color": color, "linewidth": 1.25})
        if G["group"] == "matrix":   # from the stored percentiles; no points
            box = ax.bxp([pick["box"]], positions=[i], widths=0.5, patch_artist=True, showfliers=False, **style)
        else:
            box = ax.boxplot([pick["values"]], positions=[i], widths=0.5, patch_artist=True, showfliers=False, **style)
        for artists in box.values():
            for artist in artists:
                artist.set_clip_path(clip)
        points = pick["values"]
        if len(points):
            jitter = np.random.default_rng(20261005 + i).uniform(-0.11, 0.11, len(points))
            ok = points <= Y_CAP
            ax.scatter(i + jitter[ok], points[ok], s=22, color=color, alpha=0.8, edgecolor="white", linewidth=0.4,
                       zorder=4)
            if (~ok).any():
                ax.scatter(i + jitter[~ok], np.full((~ok).sum(), FAIL_MARK_Y), s=40, marker=FAIL_MARKER, color=color,
                           edgecolor="white", linewidth=0.5, zorder=6)
        elif pick["median_um"] > Y_CAP:
            ax.scatter([i], [FAIL_MARK_Y], s=40, marker=FAIL_MARKER, color=color, edgecolor="white", linewidth=0.5,
                       zorder=6)
        y = fail_text_y(method, in_band) if method in in_band else top_of(pick) + 0.15
        ax.text(i, y, star_label(pick), ha="center", va="bottom", fontsize=9, fontweight="bold", color=color,
                zorder=7, path_effects=OUTLINE)
    ax.set_xticks(range(1, len(shown) + 1),
                  [condition_label(m, best[m]["condition"], newline=True) for m in shown])
    ax.set_xlim(0.4, len(shown) + 0.6)
    ax.set_ylabel(G["box_axis"])
    heading(fig, f"{title}: best configuration per method")
    fig.subplots_adjust(top=0.9, bottom=0.17)
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, out / f"{stem}_{G['metric']}_boxplot")


def place_labels(ax, labels, fontsize: float = 7) -> None:
    """Resolution-cap labels beside their points, on the first side that is clear of every marker
    and of the labels already placed (else the side with the fewest collisions). Works in points."""
    ax.figure.canvas.draw()                  # fix the axis limits before measuring
    pt_per_px = 72.0 / ax.figure.dpi
    points = []
    for x, y, _text, _color, area_pt2 in labels:
        px, py = ax.transData.transform((x, y))
        points.append((px * pt_per_px, py * pt_per_px, float(np.sqrt(area_pt2)) / 2))
    boxes = []
    frame = ax.get_window_extent()           # the plot area, in points: a label must stay inside it
    fx0, fy0, fx1, fy1 = (v * pt_per_px for v in (frame.x0, frame.y0, frame.x1, frame.y1))

    def collisions(box, own) -> int:
        x0, y0, x1, y1 = box
        hits = 100 * (x0 < fx0 or y0 < fy0 or x1 > fx1 or y1 > fy1)
        hits += sum(not (x1 + 1 < a0 or a1 + 1 < x0 or y1 + 1 < b0 or b1 + 1 < y0) for a0, b0, a1, b1 in boxes)
        for i, (cx, cy, rc) in enumerate(points):
            if i != own and np.hypot(cx - min(max(cx, x0), x1), cy - min(max(cy, y0), y1)) < rc + 1:
                hits += 1
        return hits

    for i, (x, y, text, color, _area) in enumerate(labels):
        cx, cy, rc = points[i]
        r = rc + 1.5                         # marker radius plus a gap
        w, h = 4.0 * len(text) * fontsize / 7, 6.0 * fontsize / 7   # text extent
        candidates = ((r, -r - h), (r, r), (-r - w, -r - h), (-r - w, r), (-w / 2, r + 1), (-w / 2, -r - h - 1))
        scored = [(collisions((cx + dx, cy + dy, cx + dx + w, cy + dy + h), i), k, dx, dy)
                  for k, (dx, dy) in enumerate(candidates)]
        _, _, dx, dy = min(scored)           # first clear side, else the least crowded one
        boxes.append((cx + dx, cy + dy, cx + dx + w, cy + dy + h))
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=color, fontsize=fontsize,
                    clip_on=True, zorder=8)


NO_RESULT = f"Fail (> {Y_CAP:g} µm or no result)"   # line figures also show runs that returned no transformation
FAIL_LEVEL_STEP = 0.45  # µm between rows in the failure band, from FAIL_MARK_Y up ...
FAIL_ROW_TOP = Y_TOP - 1.15   # ... to at most here, so the band label above stays clear


def band_levels(points, x_span: float) -> dict:
    """Rows in the failure band for points (key, x, marker area): sorted by x, each takes the lowest
    row that holds no point closer than its own and its neighbour's marker radius (in x, as a
    fraction of the axis span, about 0.6 % per point of radius). Rows are FAIL_LEVEL_STEP apart,
    or closer when more rows are needed than fit below FAIL_ROW_TOP. Returns {key: y}."""
    taken, level_of = [], {}
    for key, x, size in sorted(points, key=lambda p: p[1]):
        radius = np.sqrt(size) / 2
        level = next((i for i, row in enumerate(taken)
                      if all(abs(x - x2) > (radius + r2) * 0.006 * x_span for x2, r2 in row)), len(taken))
        if level == len(taken):
            taken.append([])
        taken[level].append((x, radius))
        level_of[key] = level
    step = FAIL_LEVEL_STEP if len(taken) < 2 else min(FAIL_LEVEL_STEP, (FAIL_ROW_TOP - FAIL_MARK_Y) / (len(taken) - 1))
    return {key: FAIL_MARK_Y + level * step for key, level in level_of.items()}


def draw_families(ax, rows, methods, x_of, annotate: bool, sizes=None, xerr_of=None):
    """One line per preset family; markers point by point. A failed median, or a run that returned
    no transformation, is drawn in the failure band with its configuration's own marker; markers
    there that would overlap are spread over rows of the band (band_levels).
    Returns (legend handles, pending cap labels)."""
    handles, labels = [], []
    plotted = []
    for family in families(rows, methods):
        frows = sorted((r for r in rows if r["family"] == family and r["tier"] and x_of(r) is not None),
                       key=lambda r: TIERS.index(r["tier"]))
        if frows:
            plotted.append((family, frows))
    xs_all = [x_of(r) for _, frows in plotted for r in frows]
    span = (max(xs_all) - min(0, min(xs_all))) if xs_all else 1.0
    size_of = lambda r, style: sizes(r) if sizes else style["markersize"] ** 2  # noqa: E731
    band = band_levels([((family, i), x_of(r), size_of(r, family_style(family)))
                        for family, frows in plotted for i, r in enumerate(frows) if r["median_um"] > Y_CAP],
                       span or 1.0)
    for family, frows in plotted:
        style = family_style(family)
        xs = [x_of(r) for r in frows]
        ys = [band[(family, i)] if r["median_um"] > Y_CAP else r["median_um"] for i, r in enumerate(frows)]
        ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"], zorder=4)
        handles.append(plt.Line2D([], [], label=family_label(family), **style))
        for x, y, r in zip(xs, ys, frows):
            if xerr_of and xerr_of(r):   # spread over the time series
                ax.errorbar(x, y, xerr=xerr_of(r), fmt="none", ecolor=style["color"], elinewidth=0.9, capsize=2.5,
                            zorder=5)
            ax.scatter(x, y, marker=style["marker"], s=size_of(r, style), facecolor=style["markerfacecolor"],
                       edgecolor=style["markeredgecolor"], linewidth=style["markeredgewidth"], zorder=6)
            if annotate and r["median_um"] <= Y_CAP:
                labels.append((x, y, r["tier"], style["color"], size_of(r, style)))
    return handles, labels


def plot_resolution(rows, methods, title, out, stem) -> list[Path]:
    fig, ax = wide_figure()
    accuracy_bands(ax, fail_text=NO_RESULT)
    handles = []
    pharaoh = next((r for r in rows if r["method"] == "pharaoh"), None)
    if pharaoh is not None:
        handles.append(ax.axhline(shown_y(pharaoh["median_um"]), color=COLORS["pharaoh"], linewidth=2.2, zorder=3,
                                  label=f"PHARAOH ({star_label(pharaoh)} µm)"))
    family_handles, _ = draw_families(ax, rows, methods, lambda r: TIERS.index(r["tier"]), annotate=False)
    handles += family_handles
    ax.set_xticks(range(len(TIERS)), TIERS)
    ax.set_xlim(-0.35, len(TIERS) - 0.65)
    ax.set_xlabel("Working resolution cap")
    ax.set_ylabel(G["median_axis"])
    fig.suptitle(f"{G['what']} versus resolution cap", x=0.5, y=0.965, fontsize=15)
    ax.spines[["top", "right"]].set_visible(False)
    base = out / f"{stem}_resolution_{G['metric']}"
    return save_page(fig, base) + save_legend(base, [{"handles": handles, "ncol": legend_columns(len(handles))}])


def plot_cost(rows, methods, title, out, stem, metric: str, time_label: str) -> list[Path]:
    is_memory = metric == "memory"
    key = "peak_gib" if is_memory else "seconds"
    fig, ax = wide_figure()
    accuracy_bands(ax, fail_text=NO_RESULT)
    xerr = None if is_memory else (lambda r: r.get("seconds_sd"))
    handles, labels = draw_families(ax, rows, methods, lambda r: r[key], annotate=True, xerr_of=xerr)
    pharaoh = next((r for r in rows if r["method"] == "pharaoh" and r[key] is not None), None)
    if pharaoh is not None:
        if xerr and xerr(pharaoh):
            ax.errorbar(pharaoh[key], shown_y(pharaoh["median_um"]), xerr=xerr(pharaoh), fmt="none",
                        ecolor=COLORS["pharaoh"], elinewidth=0.9, capsize=2.5, zorder=6)
        handles.insert(0, ax.scatter(pharaoh[key], shown_y(pharaoh["median_um"]), marker="*", s=170,
                                  color=COLORS["pharaoh"], edgecolor="white", linewidth=0.8, zorder=7,
                                  label="PHARAOH"))
    ax.set_xlim(left=0)   # time and memory on linear axes from 0
    place_labels(ax, labels)
    ax.set_xlabel("Peak memory, RSS + GPU (GiB)" if is_memory else time_label)
    ax.set_ylabel(G["median_axis"])
    fig.suptitle(f"{G['what']} versus {'peak memory' if is_memory else time_label.split(' (')[0].lower()}",
                 x=0.5, y=0.965, fontsize=15)
    ax.spines[["top", "right"]].set_visible(False)
    base = out / f"{stem}_{'memory' if is_memory else 'runtime'}_{G['metric']}"
    return save_page(fig, base) + save_legend(base, [{"handles": handles, "ncol": legend_columns(len(handles))}])


def area(peak_gib: float) -> float:
    """Marker area proportional to peak memory (RSS + GPU)."""
    return 11.0 * peak_gib


BIG = 1.5   # the runtime-with-memory figure: every font 1.5 x the other figures


def plot_runtime_with_memory(rows, methods, title, out, stem, time_label: str) -> list[Path]:
    rows = [r for r in rows if r["seconds"] is not None and r["peak_gib"] is not None]
    fig, ax = wide_figure()
    fig.subplots_adjust(top=0.87)   # room for the larger title
    accuracy_bands(ax, fontsize=9 * BIG, fail_text=NO_RESULT)
    ax.tick_params(labelsize=9 * BIG)
    family_handles, labels = draw_families(ax, rows, methods, lambda r: r["seconds"], annotate=True,
                                           sizes=lambda r: area(r["peak_gib"]), xerr_of=lambda r: r.get("seconds_sd"))
    handles = [plt.Line2D([], [], **family_style(f))   # line and marker shape; the size key is separate
               for f in families(rows, methods)]
    names = [h.get_label() for h in family_handles]
    ax.set_xlim(left=0)   # time on a linear axis
    place_labels(ax, labels, fontsize=7 * BIG)
    pharaoh = next((r for r in rows if r["method"] == "pharaoh"), None)
    if pharaoh is not None:
        if pharaoh.get("seconds_sd"):
            ax.errorbar(pharaoh["seconds"], shown_y(pharaoh["median_um"]), xerr=pharaoh["seconds_sd"], fmt="none",
                        ecolor=COLORS["pharaoh"], elinewidth=0.9, capsize=2.5, zorder=6)
        handles.insert(0, ax.scatter(pharaoh["seconds"], shown_y(pharaoh["median_um"]), s=area(pharaoh["peak_gib"]),
                                  marker="o", color=COLORS["pharaoh"], edgecolor="white", linewidth=0.8, zorder=7))
        names.insert(0, "PHARAOH")
    levels = (2, 8, 16, 32)   # GiB; RegWSI with its GPU memory reaches ~33
    sizes = [plt.scatter([], [], s=area(g), marker="o", color="#6B7280", edgecolor="white", linewidth=0.8)
             for g in levels]
    ax.add_artist(ax.legend(sizes, [f"{g} GiB" for g in levels], title="Peak memory", loc="center right",
                            frameon=False, labelspacing=0.9, fontsize=8 * BIG, title_fontsize=10 * BIG))
    ax.set_xlabel(time_label, fontsize=11 * BIG)
    ax.set_ylabel(G["median_axis"], fontsize=11 * BIG)
    fig.suptitle(f"{G['what']} versus {time_label.split(' (')[0].lower()}", x=0.5, y=0.965, fontsize=15 * BIG)
    ax.spines[["top", "right"]].set_visible(False)
    base = out / f"{stem}_runtime_{G['metric']}_with_size_memory"
    entries = {"handles": handles, "labels": names, "handlelength": 3.2, "fontsize": 8 * BIG}
    return (save_page(fig, base)   # the memory size key stays in the figure, top right
            + save_legend(base, [{**entries, "ncol": legend_columns(len(handles)), "columnspacing": 1.5}])
            + save_legend(base, [{**entries, "ncol": 1, "labelspacing": 1.4}], suffix="_legend_vertical"))


def write_extras(rows, dataset, methods, series, time_key, out, stem, figures, time_series=None) -> list[Path]:
    table = out / f"{stem}_conditions.csv"
    with table.open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["method", "method_label", "condition", "family", "resolution_cap", f"median_{G['metric']}_um",
                    f"n_over_{Y_CAP:g}um", "n_points", "complete", "run_failed", "time_sec", "time_sd_sec", "n_time_runs",
                    "time_kind", "peak_memory_gib"])
        for r in sorted(rows, key=lambda r: (methods.index(r["method"]), r["condition"])):
            w.writerow([r["method"], LABELS[r["method"]], r["condition"], r["family"], r["tier"] or "",
                        "" if r["failed"] else r["median_um"], r["n_over"], r["n_points"],
                        r["complete"], r["failed"], r["seconds"], r["seconds_sd"], r["n_time"],
                        "wall_seconds_until_stop" if r["failed"] else ("method_sec" if time_key == "method" else "wall_seconds"),
                        r["peak_gib"]])
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "series": series,
        "dataset": dataset, "methods": methods, "time": "method_sec" if time_key == "method" else "wall_seconds",
        "time_series": time_series, "time_statistic": "mean ± SD over time_series" if time_series else "one run",
        "failure_threshold_um": Y_CAP, "box_plot_selection": "lowest median among fully scored, non-matched runs",
        "files": [p.name for p in figures] + [table.name]}, indent=2) + "\n")
    readme = out / "README.md"
    if G["group"] == "matrix":
        m = G["metric"]
        readme.write_text(
            f"# {label(dataset).replace(chr(10), ' ')}: matrix figures ({series})\n\n"
            "Generated by `eval/visualization_dataset_keypoints.py --group matrix` from the matrix evaluations in "
            f"`results/{series}/matrix/<method>/`. Methods: {', '.join(LABELS[x] for x in methods)}.\n\n"
            "The metric is each run's distance of every transcript to the position 10x's alignment matrix gives "
            "(agreement with 10x, not accuracy against a ground truth).\n\n"
            f"- `{stem}_{m}_boxplot.*`: each method's lowest-median condition (post hoc); box p25 / median / p75, "
            "whiskers p5 / p95, from the stored percentiles (p5 from the 0.1 µm histogram); no points.\n"
            f"- `{stem}_resolution_{m}.*`, `{stem}_runtime_{m}.*`, `{stem}_memory_{m}.*`, "
            f"`{stem}_runtime_{m}_with_size_memory.*`: the median against resolution cap, "
            f"{'method time' if time_key == 'method' else 'wall-clock time'}, peak memory (RSS + GPU), and time with "
            "point area = peak memory; as the keypoint figures.\n"
            f"- `{stem}_conditions.csv`: every plotted run (`n_over_{Y_CAP:g}um` / `n_points` count transcripts).\n\n"
            f"`*`: at least one transcript of that run is above {Y_CAP:g} µm. A run that returned no transformation is "
            "drawn as a cross in the failure band.\n" + (f"\n{flip_note([dataset])}\n" if flip_note([dataset]) else ""))
        return [table, manifest_path, readme]
    readme.write_text(
        f"# {label(dataset).replace(chr(10), ' ')}: keypoint figures ({series})\n\n"
        "Generated by `eval/visualization_dataset_keypoints.py` from the keypoint evaluations in "
        f"`results/{series}/keypoints/<method>/`. Methods: {', '.join(LABELS[m] for m in methods)}.\n\n"
        f"- `{stem}_tre_boxplot.*`: every keypoint TRE of each method's lowest-median, fully scored, non-matched "
        "condition (post hoc, against the ground truth).\n"
        f"- `{stem}_resolution_tre.*`: median TRE across the resolution caps; one line per preset family "
        "(RegWSI initial / nonrigid / hi-res, VALIS rigid / nonrigid / micro); PHARAOH as a horizontal line.\n"
        f"- `{stem}_runtime_tre.*`, `{stem}_memory_tre.*`: median TRE against "
        f"{'method time (method_sec)' if time_key == 'method' else 'wall-clock time'} and against peak memory: "
        "the process tree's RSS plus the GPU memory the method recorded (RegWSI on MPS; unified-memory GPU "
        "buffers barely reach RSS).\n"
        f"- `{stem}_runtime_tre_with_size_memory.*`: median TRE against time, point area = peak memory (2:1 page).\n"
        f"- `{stem}_conditions.csv`: every plotted run.\n\n"
        + (f"Time: the mean over {', '.join(time_series)} (run records, via the summary "
           f"`results/summary/{summary_name([series], time_series)}/keypoints/`), ± SD as horizontal bars; "
           "`n_time_runs` in the table says how many runs each mean has. Accuracy and memory: "
           f"{series}.\n\n" if time_series else "")
        + "A run that returned no transformation (stopped by a resource guard, killed, or a registration error) is "
        "drawn as a cross in the failure band too: at its resolution cap, at the wall time until it stopped, and at its "
        "peak memory until then (`run_failed` in the table).\n\n"
        + f"The dashed line at {Y_CAP:g} µm marks failure; failed medians are drawn as crosses in the band above it, "
        "in the method's colour. `*`: at least one keypoint of that run is above the failure line.\n")
    return [table, manifest_path, readme]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--series", required=True, help="e.g. iter1")
    p.add_argument("--group", choices=tuple(GROUPS), default="keypoints",
                   help="keypoints (default) or matrix: which datasets and evaluation")
    p.add_argument("--datasets", nargs="+", help="default: every dataset of the group")
    p.add_argument("--time", choices=("method", "wall"), default="method",
                   help="time axis: method_sec (default; the charged method time) or the wall-clock time")
    p.add_argument("--methods", nargs="+", choices=METHODS,
                   help="draw only these methods into the main folder, files suffixed _without_<left-out methods>")
    p.add_argument("--time-series", nargs="+",
                   help="time = mean (± SD) over these series, e.g. iter1 iter2 iter3, from the summary of "
                        "eval/summarize.py --series <series> --time-series ...; accuracy and memory stay from --series")
    args = p.parse_args()
    G.update(GROUPS[args.group], group=args.group)
    group = args.group
    env = environment()
    results, runs_root = Path(env["results_root"]), Path(env["runs_root"])
    datasets = alphabetical([e["name"] for e in manifest.load(group, args.datasets)])
    name = summary_name([args.series], args.time_series)
    timing = None
    if args.time_series:
        table = results / "summary" / name / group / "conditions.csv"
        if not table.exists():
            raise SystemExit(f"no averaged summary at {table}; run first:\n  eval/eval_summary.sh --mode {group} "
                             f"--series {args.series} --time-series {' '.join(args.time_series)}")
        timing = {(r["method"], r["dataset"], r["condition"]): r for r in csv.DictReader(table.open())}
    base = results / "visualization" / name / group
    if args.methods:
        left_out = [m for m in METHODS if m not in args.methods]
        views = [(base, args.methods, ("_without_" + "_".join(left_out)) if left_out else "")]
    else:
        views = [(base, list(MAIN_METHODS), ""), (base / "ablation_bioformats", list(METHODS), "")]
    time_label = "Method time (s)" if args.time == "method" else "Wall-clock time (s)"
    for view_root, methods, suffix in views:
        use_labels(methods)
        for dataset in datasets:
            rows = load_runs(results / args.series / group, runs_root, args.series, dataset, methods, args.time,
                             timing)
            if not rows:
                print(f"[visualization] {dataset}: no scored runs; skipped", flush=True)
                continue
            out, stem, title = view_root / dataset, f"{dataset}{suffix}", label(dataset).replace("\n", " ")
            figures = plot_boxplot(rows, methods, title, out, stem)
            figures += plot_resolution(rows, methods, title, out, stem)
            figures += plot_cost(rows, methods, title, out, stem, "time", time_label)
            figures += plot_cost(rows, methods, title, out, stem, "memory", time_label)
            figures += plot_runtime_with_memory(rows, methods, title, out, stem, time_label)
            write_extras(rows, dataset, methods, args.series, args.time, out, stem, figures, args.time_series)
            print(f"[visualization] {dataset}: {sum('_legend' not in p.stem for p in figures) // 2} figures "
                  f"(+ legends) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
