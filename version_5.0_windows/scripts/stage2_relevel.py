"""What happens after Stage 2's automatic alignment fails.

Two independent axes, which the CLI keeps separate because they fail for different
reasons and are switched off for different purposes:

* **relevel** -- retry Stage 1/2 at other pyramid levels
  (`--stage2-relevel-rounds N`, 0 to skip);
* **manual** -- hand the pair to the interactive GUI, or in a batch to
  `manual_alignment_queue.json` (`--stage2-nomanual` to skip).

`--stage2-nofallback` is the shorthand for switching off both.

The rest of this module is the relevel ladder: which Stage-1 level pairs to retry,
and in what order.

Shared by `run_pharaoh.py` and `run_pharaoh_parallel.py`, which used to carry a copy
of this each (PLAN_4.8.md section 1, item 4).

The order is **finer first, then coarser**, one of each per round. Both directions are
tried because Stage 2 can fail for opposite reasons, and the cheap diagnosis is to try
both:

* **Finer answers undersampling.** A pair whose Stage-1 levels landed at the coarse end
  of the size window can carry too few keypoints for either detector to reach the gate,
  and one level down quadruples the pixels available to them.

* **Coarser answers noise.** Noise is independent per pixel and tissue structure is not,
  so averaging four pixels into one improves the signal-to-noise ratio by about 2x while
  leaving the architecture Stage 2 matches on -- outlines, glands, vessel walls, hundreds
  of micrometres across -- intact.

**The measurements available argue for the opposite order, and are recorded here rather
than acted on.** On `geo_lung_l7` a finer-only ladder drove ORB's ratio-test pass rate
from 0.16% to 0.07% and collapsed every fit to 4-7 inliers, the minimum a homography
needs (PLAN_4.8.md section 3); VoltRon at its 16k tier found *fewer* SIFT keypoints on
that slide than at 8k, 7,765 against 8,561, for the same reason. On `mouse_brain_S2` the
coarser rung produced the best candidate of the three levels tried -- SIFT 22/30 inliers,
ratio 0.73, 0.61 px -- while the finer rung produced 3/21 and 5/16.

Order decides only which is tried first; both are tried, so the cost of leading with the
weaker direction is one attempt of about a second.

Stage 2 matches tissue architecture; nuclei are Stage 4b's business. Detail below the
scale of what is being matched is noise to this stage, which is why the ladder leads
with the direction that removes it.
"""

from __future__ import annotations


_LEGACY_STAGE2_FLAGS = {
    "legacy_relevel_attempts": "--alignment-recovery-max-attempts",
    "legacy_relevel_rounds": "--alignment-recovery-rounds",
    "legacy_relevel_off": "--no-alignment-recovery",
    "legacy_relevel_on": "--alignment-recovery",
}


def resolve_stage2_fallback(args, error, prefix):
    """Fold the shorthand and the superseded spellings into the two real settings.

    Leaves `args.stage2_relevel_rounds` (>= 0) and `args.stage2_nomanual` holding what
    the run should actually do, and calls `error` on a rejected value. The superseded
    flags are applied first so an explicit new flag alongside one still wins.
    """
    for attribute, flag in _LEGACY_STAGE2_FLAGS.items():
        if not getattr(args, attribute, None):
            continue
        if attribute == "legacy_relevel_attempts":
            # Two attempts used to mean one level each way, i.e. one round.
            args.stage2_relevel_rounds = max(1, (args.legacy_relevel_attempts + 1) // 2)
            replacement = f"--stage2-relevel-rounds {args.stage2_relevel_rounds}"
        elif attribute == "legacy_relevel_rounds":
            args.stage2_relevel_rounds = args.legacy_relevel_rounds
            replacement = f"--stage2-relevel-rounds {args.stage2_relevel_rounds}"
        elif attribute == "legacy_relevel_off":
            # It never suppressed manual alignment, so it maps to rounds 0 alone.
            args.stage2_relevel_rounds = 0
            replacement = "--stage2-relevel-rounds 0"
        else:
            replacement = f"--stage2-relevel-rounds {args.stage2_relevel_rounds}"
        print(f"[{prefix}] {flag} is superseded by {replacement}", flush=True)

    if getattr(args, "stage2_nofallback", False):
        args.stage2_relevel_rounds = 0
        args.stage2_nomanual = True
    if args.stage2_relevel_rounds < 0:
        error("--stage2-relevel-rounds cannot be negative")


def relevel_pairs(he_level, moving_level, rounds,
                         he_max_level=None, moving_max_level=None):
    """Return `[(he_level, moving_level), ...]` to retry, in order.

    Round *k* offers `level - k` (finer) and then `level + k` (coarser). Each image is
    clamped independently -- to 0 and, when the pyramid depth is known, to its top
    level -- and a pair equal to the starting one, or already offered, is dropped. The
    result may therefore be shorter than `2 * rounds`, and empty when the pyramids
    leave nowhere to go.
    """
    try:
        he_level = int(he_level)
        moving_level = int(moving_level)
        rounds = int(rounds)
    except (TypeError, ValueError):
        return []
    if he_level < 0 or moving_level < 0 or rounds < 1:
        return []

    def clamp(level, offset, top):
        stepped = level + offset
        if stepped < 0:
            return 0
        if top is not None and stepped > top:
            return int(top)
        return stepped

    start = (he_level, moving_level)
    pairs, seen = [], {start}
    for step in range(1, rounds + 1):
        for offset in (-step, step):          # finer, then coarser
            candidate = (
                clamp(he_level, offset, he_max_level),
                clamp(moving_level, offset, moving_max_level),
            )
            if candidate in seen:
                continue
            seen.add(candidate)
            pairs.append(candidate)
    return pairs


def pyramid_top_levels(info, moving_key, fixed_key="HE_level"):
    """Highest usable pyramid level for the fixed and moving images, or (None, None).

    Read from the images themselves rather than assumed, so a retry never asks for a
    level that does not exist. Failure is not an error: the caller clamps to 0 only.
    """
    try:
        from my_utils import inspect_image_pyramid
    except Exception:
        return None, None

    def top(path):
        if not path:
            return None
        try:
            shapes = inspect_image_pyramid(path).get("level_shapes") or []
        except Exception:
            return None
        return len(shapes) - 1 if shapes else None

    return (top(info.get(fixed_key.replace("_level", "_path"))),
            top(info.get(moving_key.replace("_level", "_path"))))
