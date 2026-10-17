"""Render a parallel batch as a timeline: which dataset ran when, and in what stage.

`monitor_memory_time.svg` answers "how much memory did this dataset use"; nothing
answered "what was the machine doing at t=90s". That is the question the Stage-3 and
Stage-4 gates exist to shape, so it is the one worth being able to see: a queue wait
looks exactly like slow work in a log, and only a timeline separates them.

Reads what `--monitor` already records -- each run's stage-labelled samples plus the
batch's per-dataset start times -- and writes `batch_timeline.html` into the output
root. Also runnable on any finished batch:

    python scripts/batch_timeline.py outputs/all9
"""

from __future__ import annotations

import csv
import html
import json
from datetime import datetime
from pathlib import Path

# Stage labels are grouped rather than shown one colour each: there are 13 of them
# across a batch, well past the 8 a categorical palette can keep apart, and the
# groups are what the question is actually about. Order here is the legend order,
# which is what the palette's adjacent-pair validation covers.
GROUPS = [
    ("setup", "Stage 1-2 - read, masks, alignment"),
    ("tiles", "Stage 3 - tiles"),
    ("nuclei", "Stage 4 - nuclei"),
    ("downstream", "Stage 5-8 - pairs, refinement, outputs"),
    ("waiting", "Waiting for a memory slot"),
]
GROUP_INDEX = {key: i for i, (key, _) in enumerate(GROUPS)}

# Categorical slots 1-4 of the validated palette, plus a deliberate neutral for
# waiting. The neutral is below the chroma floor on purpose -- it should read as
# "not working" -- so it carries a diagonal hatch as secondary encoding and never
# relies on colour alone.
LIGHT = {"setup": "#2a78d6", "tiles": "#eb6834", "nuclei": "#1baf7a",
         "downstream": "#eda100", "waiting": "#a3a29b"}
DARK = {"setup": "#3987e5", "tiles": "#d95926", "nuclei": "#199e70",
        "downstream": "#c98500", "waiting": "#6c6b65"}


def stage_group(label: str) -> str:
    """Map a monitor stage label onto one of the five groups."""
    text = (label or "").strip().casefold()
    if not text or text == "startup":
        return "setup"
    if "waiting" in text or text.startswith("stage 3q") or text.startswith("stage 4q"):
        return "waiting"
    if text.startswith("stage 3"):
        return "tiles"
    if text.startswith("stage 4"):
        return "nuclei"
    if text.startswith("stage 1") or text.startswith("stage 2"):
        return "setup"
    return "downstream"


def read_spans(run_dir: Path, offset: float, end: float):
    """Contiguous [start, end, group] spans for one dataset, in batch time.

    A sample carries the stage that was active when it was taken, so a span runs to
    the next sample's timestamp. The final span is closed at the dataset's own
    finish time rather than its last sample, which would otherwise leave a gap the
    width of one sampling interval.
    """
    path = run_dir / "monitor_memory_time.csv"
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for record in csv.DictReader(handle):
            try:
                rows.append((float(record["elapsed_seconds"]),
                             stage_group(record.get("stage", ""))))
            except (TypeError, ValueError):
                continue
    if not rows:
        return []
    rows.sort()
    spans = []
    for i, (t, group) in enumerate(rows):
        stop = rows[i + 1][0] if i + 1 < len(rows) else max(t, end - offset)
        if stop <= t:
            continue
        if spans and spans[-1][2] == group and abs(spans[-1][1] - (offset + t)) < 1e-6:
            spans[-1][1] = offset + stop
        else:
            spans.append([offset + t, offset + stop, group])
    return spans


def load_batch(output_root: Path):
    summary = json.loads((output_root / "parallel_summary.json").read_text())
    t0 = datetime.fromisoformat(summary["started_at"])
    datasets = []
    for entry in summary.get("datasets", []):
        started, finished = entry.get("started_at"), entry.get("finished_at")
        if not started or not finished:
            continue
        offset = (datetime.fromisoformat(started) - t0).total_seconds()
        end = (datetime.fromisoformat(finished) - t0).total_seconds()
        run_dir = Path(entry.get("run_dir") or (output_root / f"runs_{entry['name']}"))
        datasets.append({
            "name": entry["name"],
            "start": offset,
            "end": end,
            "returncode": entry.get("returncode"),
            "spans": read_spans(run_dir, offset, end),
        })
    datasets.sort(key=lambda d: (d["start"], d["name"]))
    return summary, datasets


def concurrency(datasets, total, step=1.0):
    """Datasets doing work, and datasets waiting, sampled across the batch."""
    points = []
    t = 0.0
    while t <= total:
        working = waiting = 0
        for d in datasets:
            for start, stop, group in d["spans"]:
                if start <= t < stop:
                    if group == "waiting":
                        waiting += 1
                    else:
                        working += 1
                    break
        points.append((t, working, waiting))
        t += step
    return points


def _ticks(total):
    for candidate in (10, 15, 20, 30, 60, 120, 300, 600):
        if total / candidate <= 12:
            step = candidate
            break
    else:
        step = 900
    values, t = [], 0
    while t <= total:
        values.append(t)
        t += step
    return values


def _fmt(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}" if seconds >= 60 else f"{seconds}s"


def _page(*, title, subtitle, legend, width, height, marks, table):
    """The page skeleton: tokens, both dark-mode scopes, legend, table view.

    Kept separate from the chart body so the theming contract -- light tokens on
    bare :root, dark ones under both the media query and the data-theme scope --
    lives in one place rather than tangled through the mark generation.
    """
    return f"""<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
  .viz {{
    color-scheme: light;
    --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
    --grid: #e4e3df; --axis: #b9b8b2; --hatch-ink: #a3a29b; --hatch-swatch: #d8d7d1; --strip-ink: #8f8e87;
    --c-setup: #2a78d6; --c-tiles: #eb6834; --c-nuclei: #1baf7a;
    --c-downstream: #eda100; --c-waiting: #a3a29b;
    background: var(--surface-1); color: var(--text-primary);
    font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif;
    padding: 20px; max-width: 1180px; margin: 0 auto;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz {{
      color-scheme: dark;
      --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
      --grid: #2f2f2c; --axis: #55544f; --hatch-ink: #6c6b65; --hatch-swatch: #3a3a36; --strip-ink: #86857d;
      --c-setup: #3987e5; --c-tiles: #d95926; --c-nuclei: #199e70;
      --c-downstream: #c98500; --c-waiting: #6c6b65;
    }}
  }}
  :root[data-theme="dark"] .viz {{
    color-scheme: dark;
    --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --grid: #2f2f2c; --axis: #55544f; --hatch-ink: #6c6b65; --hatch-swatch: #3a3a36; --strip-ink: #86857d;
    --c-setup: #3987e5; --c-tiles: #d95926; --c-nuclei: #199e70;
    --c-downstream: #c98500; --c-waiting: #6c6b65;
  }}
  body {{ margin: 0; background: var(--surface-1); }}
  h1 {{ font-size: 17px; margin: 0 0 2px; font-weight: 600; }}
  .sub {{ color: var(--text-secondary); font-size: 13px; margin: 0 0 14px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 0 0 10px; font-size: 13px;
             color: var(--text-secondary); }}
  .key {{ display: inline-flex; align-items: center; gap: 7px; }}
  .key i {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
  .key i.hatch {{ background: repeating-linear-gradient(45deg,
      var(--hatch-swatch) 0 3px, var(--hatch-ink) 3px 5px); }}
  .scroll {{ overflow-x: auto; }}
  svg {{ display: block; min-width: 1154px; }}
  text {{ font: 12px ui-sans-serif, -apple-system, system-ui, sans-serif; fill: var(--text-secondary); }}
  .row {{ fill: var(--text-primary); font-size: 12.5px; }}
  .dur {{ fill: var(--text-secondary); font-size: 11px; }}
  .end {{ text-anchor: end; }} .mid {{ text-anchor: middle; }}
  .lbl {{ font-size: 11px; }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }}
  .axis {{ stroke: var(--axis); stroke-width: 1; }}
  .span {{ stroke: var(--surface-1); stroke-width: 0; }}
  .span:hover {{ stroke-width: 2; }}
  details {{ margin-top: 16px; font-size: 13px; color: var(--text-secondary); }}
  table {{ border-collapse: collapse; margin-top: 8px; font-size: 12.5px; }}
  th, td {{ text-align: right; padding: 3px 12px 3px 0; }}
  th:first-child, td:first-child {{ text-align: left; color: var(--text-primary); }}
  th {{ border-bottom: 1px solid var(--grid); font-weight: 600; }}
</style>
<div class="viz">
  <h1>{title}</h1>
  <p class="sub">{subtitle}</p>
  <div class="legend">{legend}</div>
  <div class="scroll">
  <svg viewBox="0 0 {width} {height}" width="{width}" height="{height}"
       role="img" aria-label="{title}">
    <defs>
      <pattern id="hatch" width="7" height="7" patternTransform="rotate(45)"
               patternUnits="userSpaceOnUse">
        <rect width="7" height="7" fill="var(--hatch-swatch)"/>
        <line x1="0" y1="0" x2="0" y2="7" stroke="var(--hatch-ink)" stroke-width="2.5"/>
      </pattern>
    </defs>
    {marks}
  </svg>
  </div>
  <details>
    <summary>Table view</summary>
    <table>{table}</table>
  </details>
</div>
"""


def render(output_root: Path) -> Path:
    summary, datasets = load_batch(output_root)
    total = float(summary.get("elapsed_seconds") or 0.0)
    if not datasets or total <= 0:
        raise SystemExit(f"no timeline data in {output_root}")

    row_h, gap, left, right, top = 26, 8, 210, 44, 22
    plot_w = 900
    strip_h = 60
    body_h = len(datasets) * (row_h + gap)
    axis_h = 52          # ticks sit at +19, the axis title at +36
    height = top + strip_h + 26 + body_h + axis_h
    width = left + plot_w + right
    x = lambda t: left + (t / total) * plot_w                       # noqa: E731

    peak = max((w + q for _, w, q in concurrency(datasets, total)), default=1) or 1
    show_strip = peak > 1
    if not show_strip:
        strip_h = 0
        height = top + body_h + axis_h
    marks = []

    # --- concurrency strip -------------------------------------------------
    # A step area, not a line: the count is piecewise-constant, and interpolating
    # between samples drew slopes that never happened. Working and waiting stack, so
    # the outline is "datasets in flight" and the split shows how much of that was
    # queue rather than work.
    strip_top = top
    strip_base = strip_top + strip_h
    if show_strip:
        points = concurrency(datasets, total, step=max(0.5, total / 600.0))
        scale = lambda v: strip_base - (v / peak) * strip_h              # noqa: E731

        def step_area(values, fill, opacity):
            """Closed path stepping through (t, value), dropped to the baseline."""
            if not values:
                return ""
            d = [f"M {x(values[0][0]):.2f} {strip_base:.2f}"]
            for i, (t, lo, hi) in enumerate(values):
                d.append(f"L {x(t):.2f} {scale(hi):.2f}")
                nxt = values[i + 1][0] if i + 1 < len(values) else total
                d.append(f"L {x(nxt):.2f} {scale(hi):.2f}")
            d.append(f"L {x(total):.2f} {strip_base:.2f} Z")
            return (f'<path d="{" ".join(d)}" fill="{fill}" opacity="{opacity}" '
                    f'stroke="none"/>')

        stacked = [(t, 0, w + q) for t, w, q in points]      # outline: total in flight
        working = [(t, 0, w) for t, w, q in points]
        marks.append(step_area(stacked, "url(#hatch)", "1"))
        marks.append(step_area(working, "var(--strip-ink)", "1"))

        for level in range(1, peak + 1):
            y = scale(level)
            marks.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
                         f'class="grid"/>')
            marks.append(f'<text x="{left - 8}" y="{y + 4:.2f}" class="tick end">{level}</text>')
        marks.append(f'<text x="{left - 8}" y="{strip_top - 8}" class="lbl end">'
                     f'datasets in flight</text>')
        marks.append(f'<text x="{left + plot_w}" y="{strip_top - 8}" class="lbl end">'
                     f'solid = working \u00b7 hatched = queued</text>')

    # --- one row per dataset ----------------------------------------------
    rows_top = (strip_base + 26) if show_strip else top
    for i, d in enumerate(datasets):
        y = rows_top + i * (row_h + gap)
        marks.append(f'<line x1="{left}" y1="{y + row_h / 2:.1f}" x2="{left + plot_w}" '
                     f'y2="{y + row_h / 2:.1f}" class="grid"/>')
        marks.append(f'<text x="{left - 10}" y="{y + row_h / 2 + 4:.1f}" '
                     f'class="row end">{html.escape(d["name"])}</text>')
        for start, stop, group in d["spans"]:
            x0, x1 = x(start), x(stop)
            w = max(1.2, x1 - x0 - 2)      # 2px surface gap between adjacent fills
            fill = "url(#hatch)" if group == "waiting" else f"var(--c-{group})"
            title = (f"{d['name']} - {dict(GROUPS)[group]}\n"
                     f"{_fmt(start)} to {_fmt(stop)}  ({stop - start:.1f}s)")
            marks.append(
                f'<rect x="{x0:.2f}" y="{y}" width="{w:.2f}" height="{row_h}" rx="4" '
                f'fill="{fill}" class="span"><title>{html.escape(title)}</title></rect>'
            )
        marks.append(f'<text x="{x(d["end"]) + 6:.2f}" y="{y + row_h / 2 + 4:.1f}" '
                     f'class="dur">{_fmt(d["end"] - d["start"])}</text>')

    # --- time axis ---------------------------------------------------------
    axis_y = rows_top + body_h + 6
    marks.append(f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_w}" y2="{axis_y}" class="axis"/>')
    for t in _ticks(total):
        marks.append(f'<line x1="{x(t):.2f}" y1="{axis_y}" x2="{x(t):.2f}" y2="{axis_y + 5}" class="axis"/>')
        marks.append(f'<text x="{x(t):.2f}" y="{axis_y + 19}" class="tick mid">{_fmt(t)}</text>')
    marks.append(f'<text x="{left + plot_w / 2:.0f}" y="{axis_y + 38}" class="lbl mid">'
                 f'time since batch start</text>')

    present = {g for d in datasets for _, _, g in d["spans"]}
    legend = "".join(
        f'<span class="key"><i style="background:{"var(--hatch-swatch)" if k == "waiting" else f"var(--c-{k})"}"'
        f'{" class=hatch" if k == "waiting" else ""}></i>{html.escape(label)}</span>'
        for k, label in GROUPS if k in present
    )

    table_rows = "".join(
        "<tr><td>{n}</td><td>{s}</td><td>{e}</td><td>{d:.1f}</td><td>{w:.1f}</td><td>{r}</td></tr>".format(
            n=html.escape(d["name"]), s=_fmt(d["start"]), e=_fmt(d["end"]),
            d=d["end"] - d["start"],
            w=sum(stop - start for start, stop, g in d["spans"] if g == "waiting"),
            r="ok" if d["returncode"] == 0 else f'exit {d["returncode"]}',
        )
        for d in datasets
    )

    jobs = summary.get("jobs")
    subtitle = (f"{len(datasets)} datasets · --jobs {jobs} · {_fmt(total)} wall clock · "
                + (f"peak {peak} in flight" if peak > 1 else "run one at a time"))

    page = _page(
        title=html.escape("Which dataset was running when"),
        subtitle=html.escape(subtitle),
        legend=legend,
        width=width, height=height,
        marks="".join(marks),
        table=('<tr><th>dataset</th><th>start</th><th>end</th><th>seconds</th>'
               '<th>of which waiting</th><th>result</th></tr>' + table_rows),
    )
    out = output_root / "batch_timeline.html"
    out.write_text(page)
    return out


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_root", type=Path,
                        help="a batch output root containing parallel_summary.json")
    args = parser.parse_args(argv)
    print(render(args.output_root.expanduser().resolve()))


if __name__ == "__main__":
    main()
