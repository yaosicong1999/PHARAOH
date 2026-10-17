"""Stage 4b anchor modes: what `stage4b.anchor_mode` may say and what it means.

Stage 4b finds one of two kinds of anchor for Stage 5:

  nuclei_pairs  the same nucleus seen on both sides, paired by overlap after a
                binary-Dice tile alignment. Each side's coordinate is measured
                independently, so this is the only anchor that carries a real
                correspondence.
  density       conserved nuclear architecture -- where nuclei crowd on both
                sides -- located on the density-NCC-aligned canvas. Adjacent
                sections share no nucleus, so pairs cannot exist there; the
                density field is what they do share.

`anchor_mode` in `parameters.json` names the ladder `run_pharaoh.py` climbs:

  auto          nuclei_pairs on the selected tiles -> top the tile supply up and
                try pairs again -> density on every Stage-3 tile -> fail
  nuclei_pairs  nuclei_pairs -> top up -> fail
  density       density on the selected tiles -> top up -> fail

Each rung is one `4b_find_standout_nuclei.py` process running one *level*
(`--anchor-mode nuclei_pairs` or `--anchor-mode density`). The launcher owns the
climb; Stage 4b never falls back on its own, and it no longer ships tile centres
when a rung is short -- it records the shortfall and fails so the launcher can
answer it. The old `correspondence_mode` key and its `same_section` /
`serial_section` values are read as aliases.
"""

MODES = ("auto", "nuclei_pairs", "density")
LEVELS = ("nuclei_pairs", "density")
DEFAULT_MODE = "auto"
PARAM_KEY = "anchor_mode"
LEGACY_PARAM_KEY = "correspondence_mode"
# `run_pharaoh.py --stage4b-anchor-mode` exports this so the Stage-4b subprocess
# sees the same effective mode the launcher used to choose its ladder.
ANCHOR_MODE_ENV_VAR = "PHARAOH_STAGE4B_ANCHOR_MODE"

_ALIASES = {
    "auto": "auto",
    "nuclei_pairs": "nuclei_pairs", "nuclei-pairs": "nuclei_pairs",
    "nuclei_pair": "nuclei_pairs", "pairs": "nuclei_pairs",
    "same_section": "nuclei_pairs", "same-section": "nuclei_pairs", "same": "nuclei_pairs",
    "density": "density", "density_ncc": "density", "architecture": "density",
    "serial_section": "density", "serial-section": "density", "serial": "density",
    "across_section": "density", "across-section": "density",
}

#: the score each level searches its tile alignment with
SCORE_METRIC = {"nuclei_pairs": "binary_dice", "density": "density_ncc"}


def normalize_mode(value):
    """Canonical `anchor_mode` for a parameters.json / command-line spelling."""
    text = str(value).strip().lower()
    if text not in _ALIASES:
        raise ValueError(
            f"stage4b.{PARAM_KEY} must be one of {', '.join(MODES)} (got {value!r})"
        )
    return _ALIASES[text]


def mode_from_block(stage4b):
    """The configured mode from a `stage4b` parameter block, aliases included."""
    if PARAM_KEY in stage4b:
        return normalize_mode(stage4b[PARAM_KEY])
    if LEGACY_PARAM_KEY in stage4b:
        return normalize_mode(stage4b[LEGACY_PARAM_KEY])
    return DEFAULT_MODE


def first_level(mode):
    """The level a mode starts at: `auto` climbs from nuclei pairs."""
    return "density" if normalize_mode(mode) == "density" else "nuclei_pairs"


def ladder(mode):
    """The rungs the launcher climbs for a mode, in order.

    Each rung is (level, scope): `selected` scores Stage 4a's spatial subset and
    can be topped up; `all_tiles` masks and scores every Stage-3 tile at once.
    """
    mode = normalize_mode(mode)
    if mode == "auto":
        return (("nuclei_pairs", "selected"), ("density", "all_tiles"))
    return ((first_level(mode), "selected"),)
