"""Shared style and helpers of the benchmark figures (eval/visualization_*.py).

One place for the method colours and labels, the accuracy bands, the failure band (above
FAILURE_UM, shared with the summary), the median labels (* = a point above the failure line)
and how figures are saved, so every figure reads the same way.
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np

from .runs import FAILURE_UM

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "benchmark_new_mpl"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402

# A thin white outline keeps a number legible over points of its own colour (no background box).
OUTLINE = [patheffects.withStroke(linewidth=2.2, foreground="white")]

METHODS = ("pharaoh", "voltron", "regwsi", "valis_tifffile", "valis_bioformats")
# The main figure leaves VALIS (Bio-Formats) out; it appears only in the ablation_bioformats view.
MAIN_METHODS = ("pharaoh", "voltron", "regwsi", "valis_tifffile")
LABELS = {"pharaoh": "PHARAOH", "voltron": "VoltRon", "regwsi": "RegWSI",
          "valis_tifffile": "VALIS (tifffile)", "valis_bioformats": "VALIS (Bio-Formats)"}
_FULL_LABELS = dict(LABELS)


def method_label(method: str, preset: str = "", cap: str = "") -> str:
    """Display name of a method, its preset and resolution cap: VALIS-micro (4k), RegWSI-hires (16k),
    VoltRon (16k), VALIS-rigid; in the ablation view VALIS-micro (Bio-Formats, 4k)."""
    name, _, variant = LABELS[method].partition(" (")
    name = f"{name}-{preset}" if preset else name
    extra = ", ".join(x for x in (variant.rstrip(")"), cap) if x)
    return f"{name} ({extra})" if extra else name


def condition_label(method: str, condition: str, newline: bool = False) -> str:
    """method_label of a benchmark condition (<preset>_<cap>, or just <cap>); PHARAOH has one
    configuration and no condition. newline=True puts the bracket on its own line."""
    if method == "pharaoh" or not condition:
        return LABELS[method]
    preset, _, cap = condition.rpartition("_")
    text = method_label(method, preset, cap)
    return text.replace(" (", "\n(", 1) if newline else text


def use_labels(methods) -> None:
    """Name the methods for one view: VALIS (tifffile) is plain "VALIS" unless VALIS (Bio-Formats)
    is drawn beside it (the ablation view). Changes LABELS in place, so everything drawn for the
    view reads the same names; call it at the start of each view."""
    LABELS.clear()
    LABELS.update(_FULL_LABELS)
    if "valis_bioformats" not in methods:
        LABELS["valis_tifffile"] = "VALIS"
# The first four are the earlier benchmark's colours; VALIS (Bio-Formats) adds cyan.
COLORS = {"pharaoh": "#15803D", "voltron": "#F59E0B", "regwsi": "#DC2626",
          "valis_tifffile": "#2563EB", "valis_bioformats": "#0891B2"}
SECONDARY = {"valis_bioformats": {"hatch": "////", "linestyle": (0, (3, 2))}}
# A method's second-best configuration, where one is drawn next to the best: a darker shade of its colour.
SECOND_COLORS = {"voltron": "#D97706", "regwsi": "#991B1B", "valis_tifffile": "#1E3A8A",
                 "valis_bioformats": "#155E75"}
FAIL_COLOR, FAIL_BAND = "#6B1E2D", "#E5E7EB"
Y_CAP, Y_TOP = FAILURE_UM, 14.0   # failure threshold (shared with the summary); the band above it shows the failures
FAIL_MARK_Y = Y_CAP + 0.35   # crosses of failed values
FAIL_MARKER = "X"            # a failed value: a cross in the method's colour, drawn in the failure band
FAIL_TEXT_Y = Y_CAP + 0.65   # first label of a failed median, just above the crosses; further ones stack upwards
FAIL_TEXT_STEP = 0.55
DATASET_LABELS = {"human_crc_p1": "Human CRC\nP1", "human_crc_p2": "Human CRC\nP2",
                  "human_crc_p5": "Human CRC\nP5", "human_kidney_ccrcc": "Human kidney\nccRCC",
                  "mouse_brain_release_1": "Mouse brain\nrelease 1",
                  "mouse_brain_release_2": "Mouse brain\nrelease 2",
                  # matrix datasets
                  "human_bone_marrow_normal": "Human bone\nmarrow", "human_femur_bone": "Human\nfemur bone",
                  "human_breast_cancer_5k": "Human breast\ncancer 5K", "human_breast_cancer_s1r1": "Human breast\ncancer S1R1",
                  "human_breast_cancer_s1r2": "Human breast\ncancer S1R2", "human_cervical_cancer": "Human cervical\ncancer",
                  "human_pancreatic_ductal_adenocarcinoma": "Human\nPDAC", "human_heart": "Human\nheart",
                  "human_liver_cancer": "Human liver\ncancer", "human_lung_cancer": "Human lung\ncancer",
                  "human_ovarian_cancer": "Human ovarian\ncancer", "human_prostate_adenocarcinoma": "Human prostate\nadenocarcinoma",
                  "human_skin_nondiseased": "Human skin\n(non-diseased)", "human_skin_melanoma": "Human skin\nmelanoma",
                  "human_tonsil_follicular_lymphoid_hyperplasia": "Human tonsil\nFLH",
                  "human_tonsil_reactive_follicular_hyperplasia": "Human tonsil\nRFH",
                  "mouse_colon": "Mouse\ncolon", "human_brain_cancer": "Human brain\ncancer", "human_crc": "Human\nCRC"}
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
                11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
                17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"}


def alphabetical(datasets: list[str]) -> list[str]:
    """The datasets in alphabetical order of their display names (as the figures print them)."""
    return sorted(datasets, key=lambda d: label(d).replace("\n", " ").lower())


def count_word(n: int) -> str:
    """A count as a word for figure titles ("six"), digits beyond twenty."""
    return NUMBER_WORDS.get(n, str(n))
ABLATIONS = ("matched",)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 10, "axes.titlesize": 14, "axes.labelsize": 11, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                     # $...$ (e.g. the italic p of a p-value) in the same Arial as the rest of the text
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold"})


# Datasets whose H&E was flipped vertically (y -> H-1-y) before registration, for every method; 10x's
# alignment matrix was adjusted to match (README, matrix datasets). Their labels carry FLIP_MARK.
FLIPPED = ("human_breast_cancer_s1r1", "human_breast_cancer_s1r2")
FLIP_MARK = "†"
FLIP_NOTE = (f"{FLIP_MARK} H&E flipped vertically (y → H−1−y) before registration for every method; "
             "10x's alignment adjusted to match.")
# the same note in a figure legend (tables and READMEs keep FLIP_NOTE)
FLIP_LEGEND = (f"{FLIP_MARK} H&E manually flipped to match the DAPI. "
               "Only PHARAOH handles the original mirrored H&E automatically.")
# its first sentence only: the matrix box plot's legend, where it is one of three note lines
FLIP_LEGEND_SHORT = f"{FLIP_MARK} H&E manually flipped to match the DAPI."


def label(dataset: str) -> str:
    """Two-line axis label of a dataset (flipped datasets end with FLIP_MARK)."""
    text = DATASET_LABELS.get(dataset, dataset.replace("_", " "))
    return text + FLIP_MARK if dataset in FLIPPED else text


def flip_note(datasets) -> str:
    """FLIP_NOTE when any of `datasets` is flipped, else ""."""
    return FLIP_NOTE if any(d in FLIPPED for d in datasets) else ""


def select(results: Path, datasets: list[str], methods: list[str]) -> dict:
    """{(dataset, method): {"condition", "median_um", "values", "source"}} for the best eligible run."""
    return {key: picks[0] for key, picks in select_ranked(results, datasets, methods, 1).items()}


def select_ranked(results: Path, datasets: list[str], methods: list[str], n: int) -> dict:
    """{(dataset, method): [pick, ...]}: the n eligible runs with the lowest median TRE, best first.
    Eligible: successful, every landmark measured, not an ablation (`matched`)."""
    ranked = {}
    for method in methods:
        table = results / method / "runs.csv"
        if not table.exists():
            print(f"[visualization] {method}: no evaluation at {table}; left out", flush=True)
            continue
        for row in csv.DictReader(table.open()):
            if (row["outcome"] != "success" or row.get("complete") != "True" or not row.get("median_um")
                    or row["condition"] in ABLATIONS or row["dataset"] not in datasets):
                continue
            ranked.setdefault((row["dataset"], method), []).append(
                {"condition": row["condition"], "median_um": float(row["median_um"])})
    for (dataset, method), picks in ranked.items():
        picks.sort(key=lambda p: p["median_um"])
        del picks[n:]
        for pick in picks:
            points = results / method / "points" / dataset / f"{pick['condition']}.csv"
            pick["values"] = np.array([float(r["tre_um"]) for r in csv.DictReader(points.open())
                                       if r["measured"] == "1"])
            pick["source"] = str(points)
    return ranked


def accuracy_bands(ax, fontsize: float = 9, fail_text: str | None = None, top: float = Y_TOP,
                   band: float = Y_TOP, fail_y: float | None = None) -> None:
    """Accuracy bands (green < 3 µm, blue 3-6, light red 6 to the failure line), the failure line
    and the grey failure band up to `band`. With `top` above the band the axis continues as a white
    strip for the test brackets; the y axis and its ticks then stop at the failure band. The band's
    label sits at its top left, or with its bottom at `fail_y` when given."""
    for low, high, color in ((0, 3, "#DFF2E1"), (3, 6, "#E2EEF8"), (6, Y_CAP, "#FBE8E8"), (Y_CAP, band, FAIL_BAND)):
        ax.axhspan(low, high, color=color, zorder=0)
    for y in (3, 6):
        ax.axhline(y, color="white", linewidth=0.8, zorder=1)
    ax.axhline(Y_CAP, color=FAIL_COLOR, linestyle=(0, (5, 4)), linewidth=1.6, zorder=2)
    ax.text(0.01, band - 0.08 if fail_y is None else fail_y, fail_text or f"Fail (> {Y_CAP:g} µm)",
            transform=ax.get_yaxis_transform(), ha="left", va="top" if fail_y is None else "bottom",
            color=FAIL_COLOR, fontsize=fontsize, fontweight="bold", zorder=7)
    ax.set_ylim(0, top)
    if top > band:
        ax.set_facecolor("white")
        ax.set_yticks(np.arange(0, Y_CAP + 0.1, 2))
        ax.spines["left"].set_bounds(0, band)


BRACKET_PT = 12.0   # vertical pitch of the nested test brackets, in points (one line of p-value text)


def bracket_step(points_per_unit: float) -> float:
    """BRACKET_PT in data units, for an axis with `points_per_unit` points per micrometre."""
    return BRACKET_PT / points_per_unit


def bracket_top(levels: int, step: float, band: float = Y_TOP) -> float:
    """Top of the y axis when `levels` nested brackets, `step` apart, sit in the white strip above
    the failure band (which ends at `band`)."""
    return band + (levels + 0.4) * step if levels else band


def pharaoh_tests(reference: np.ndarray, others: list) -> dict:
    """One-sided paired Wilcoxon signed-rank tests, PHARAOH against each configuration over the same
    keypoints (H1: PHARAOH's TRE is lower), Holm-adjusted over the family `others` (one dataset's
    comparisons). others: [(key, values)]; returns {key: {n, statistic, p, p_holm}}."""
    from scipy.stats import wilcoxon
    raw = {}
    for key, values in others:
        if len(values) != len(reference):
            raise ValueError(f"{key}: {len(values)} keypoints against PHARAOH's {len(reference)}")
        if np.all(reference == values):
            raw[key] = {"n": len(values), "statistic": 0.0, "p": 1.0}
            continue
        result = wilcoxon(reference, values, alternative="less")
        raw[key] = {"n": len(values), "statistic": float(result.statistic), "p": float(result.pvalue)}
    running = 0.0
    order = sorted(raw, key=lambda k: raw[k]["p"])
    for i, key in enumerate(order):   # Holm step-down
        running = max(running, min(1.0, (len(order) - i) * raw[key]["p"]))
        raw[key]["p_holm"] = running
    return raw


def p_label(p: float) -> str:
    """The p-value as typeset in a paper: italic p (mathtext, in the figure's Arial italic)."""
    return "$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}" if p < 0.01 else f"$p$ = {p:.2f}"


def draw_brackets(ax, x_ref: float, targets: list, fontsize: float, step: float, band: float = Y_TOP) -> None:
    """Nested brackets in the white strip above the failure band, from PHARAOH (x_ref) to each
    compared box, nearest lowest; each adjusted p-value on its bracket's line, left of PHARAOH, in
    that box's colour. targets: [(x, p, colour)]."""
    for level, (x, p, color) in enumerate(sorted(targets, key=lambda t: abs(t[0] - x_ref))):
        y = band + (level + 0.6) * step
        tick = 0.3 * step
        ax.plot([x_ref, x_ref, x, x], [y - tick, y, y, y - tick], color="#374151", linewidth=0.8,
                solid_joinstyle="miter", zorder=7, clip_on=False)
        ax.annotate(p_label(p), (x_ref, y), xytext=(-4, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=fontsize, color=color, zorder=8, annotation_clip=False)


# Marks a median whose run has at least one keypoint above the failure line. (* is kept for the
# dataset's lowest median, written *between asterisks*, and † for the flipped H&E datasets.)
OVER_MARK = "‡"


def median_label(pick: dict) -> str:
    """The median, with OVER_MARK when at least one keypoint of the run is above the failure line."""
    return f"{pick['median_um']:.2f}" + (OVER_MARK if (pick["values"] > Y_CAP).any() else "")


def box_top(points: np.ndarray) -> float:
    """Top of the drawn box: the upper whisker (matplotlib's default 1.5 IQR rule) or, when outliers
    lift it above the whisker, the box's upper edge (Q3)."""
    q1, q3 = np.percentile(points, [25, 75])
    return float(max(q3, points[points <= q3 + 1.5 * (q3 - q1)].max()))


def fail_text_y(method: str, failed: list[str]) -> float:
    """Failed medians of one dataset stack upwards in method order, so their labels never overlap."""
    return FAIL_TEXT_Y + FAIL_TEXT_STEP * failed.index(method)


def save_legend(base: Path, legends: list[dict], suffix: str = "_legend", below: bool = False) -> list[Path]:
    """The legend of a figure as its own file, <base><suffix>.{png,pdf}: each entry of `legends`
    (keyword arguments of Figure.legend: handles, labels, ncol, title, ...) side by side, or with
    below=True each under the one before, left-aligned."""
    fig = plt.figure(figsize=(1, 1))
    x, y = 0.0, 1.0
    for spec in legends:
        legend = fig.legend(loc="upper left", bbox_to_anchor=(x, y), frameon=False, **spec)
        fig.canvas.draw()
        box = legend.get_window_extent().transformed(fig.transFigure.inverted())
        if below:
            y = box.y0
        else:
            x = box.x1 + 0.25
    return save(fig, base.with_name(base.name + suffix))


BEST_NOTE = "*x.xx* Lowest median in the dataset"


def note_legend(notes: list[str]) -> dict:
    """Figure.legend keywords for lines of text without markers: the notes on a legend's marks."""
    return {"handles": [plt.Line2D([], [], linestyle="none") for _ in notes], "labels": notes, "ncol": 1,
            "handlelength": 0, "handletextpad": 0, "labelspacing": 0.3}


def save(fig, base: Path, tight: bool = True) -> list[Path]:
    """<base>.png (300 DPI) and .pdf; tight=False keeps the figure's own page size (an exact aspect
    ratio) instead of cropping to the drawn content."""
    base.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for suffix in (".png", ".pdf"):
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight" if tight else None, facecolor="white",
                    **({"dpi": 300} if suffix == ".png" else {}))
        out.append(path)
    plt.close(fig)
    return out
