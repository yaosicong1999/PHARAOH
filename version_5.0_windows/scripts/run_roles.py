# ============================================================
# run_roles.py - which image is moving, which is fixed, and what each one is
#
# 4.10 named images by modality and could not make the fixed side a choice: `--he`
# was required, `images_info.json` carried `HE_path`, and Stage 5 wrote
# `{moving}_to_he_homography_level0.json` with the `he` spelled into the name. It
# could express DAPI onto H&E and H&E onto H&E, and nothing else.
#
# 4.11 takes the pair as the input -- `--pair dapi-he --moving A --fixed B` -- and
# there are exactly two modalities, `dapi` and `he`. The pair decides how each side
# is spelled everywhere downstream:
#
#     different modalities -> the modality is the word     dapi_to_he_...
#     same modality        -> fg for moving, bg for fixed  hefg_to_hebg_...
#
# The suffix is a disambiguator, not a general role marker, which is why `dapi-he`
# comes out byte-identical to 4.10: `1_dapi_level_image.png`, `1_he_level_image.png`,
# `dapi_to_he_homography_level0.json`. Only the same-modality pairs are new spellings.
#
# `he0` is gone. It was 4.10's internal name for the moving H&E and it was one of
# four spellings for the same image -- `hefg` on the command line, `he0` inside,
# `he0fg` as the parameters key, `he` as the `--mode` token and the engine directory.
# `my_utils.cli_params_block` records the bill: looking a block up by the `he0`
# spelling returned an empty dict, because the key was `he0fg`.
#
# Nothing here imports cv2 or numpy: every stage and every test can load it.
# ============================================================
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

MOVING = "moving"
FIXED = "fixed"
ROLES = (MOVING, FIXED)

#: The only two modalities. A third would be one more entry here and its pairs below.
MODALITIES = ("dapi", "he")

#: Appended to the modality only when both sides are the same modality.
ROLE_SUFFIX = {MOVING: "fg", FIXED: "bg"}

#: The key `images_info.json` carries the binding under.
ROLES_KEY = "roles"


@dataclass(frozen=True)
class Pair:
    """One (moving, fixed) combination and everything that follows from it."""
    name: str
    moving: str
    fixed: str
    engine: str
    params_block: str
    implemented: bool
    note: str = ""

    @property
    def same_modality(self) -> bool:
        return self.moving == self.fixed

    def modality(self, role: str) -> str:
        if role == MOVING:
            return self.moving
        if role == FIXED:
            return self.fixed
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")

    def token(self, role: str) -> str:
        """The word this side is spelled with in file names and `images_info` keys.

        `fg` / `bg` appear only when the modality alone would be ambiguous, so a
        `dapi-he` run keeps every name 4.10 used and nothing downstream of it moves.
        """
        modality = self.modality(role)
        if not self.same_modality:
            return modality
        return modality + ROLE_SUFFIX[role]

    def key_prefix(self, role: str) -> str:
        return self.token(role).upper()

    @property
    def homography_stem(self) -> str:
        return f"{self.token(MOVING)}_to_{self.token(FIXED)}_homography_level0"

    def stage1_image(self, role: str) -> str:
        return f"1_{self.token(role)}_level_image.png"


PAIRS: Dict[str, Pair] = {
    "dapi-he": Pair(
        "dapi-he", "dapi", "he", "engine_dapi", "dapi-he", True,
        "Xenium DAPI onto its H&E; Stages 1-8",
    ),
    "he-dapi": Pair(
        "he-dapi", "he", "dapi", "engine_dapi", "he-dapi", True,
        "H&E onto DAPI; Stages 1-8, with H&E moving and DAPI fixed",
    ),
    "dapi-dapi": Pair(
        "dapi-dapi", "dapi", "dapi", "engine_dapi", "dapi-dapi", True,
        "DAPI onto DAPI; Stages 1-6, without assuming a section relationship",
    ),
    "he-he": Pair(
        "he-he", "he", "he", "engine_dapi", "he-he", True,
        "H&E onto H&E; Stages 1-6, without assuming a section relationship",
    ),
}

#: What `--pair` means when it is not given.
DEFAULT_PAIR = "dapi-he"

#: Spellings anyone might type. `_` for `-` is the common slip; the rest are the
#: names 4.10 used for the H&E-onto-H&E pair before `he0` and `hefg` were retired.
PAIR_ALIASES: Dict[str, str] = {
    "dapi_he": "dapi-he",
    "he_dapi": "he-dapi",
    "dapi_dapi": "dapi-dapi",
    "he_he": "he-he",
    "hefg-he": "he-he",
    "hefg_he": "he-he",
}


@dataclass(frozen=True)
class ImageRole:
    """One side of a run: what it is, where it is, and how Stage 1 read it."""
    role: str
    modality: str
    path: str
    level: int
    token: str = ""
    threshold: Optional[int] = None
    orientation_case: int = 0
    gui_affine: Optional[List[List[float]]] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def channel_index(self) -> Optional[int]:
        """Zero-based DAPI source channel; legacy runs retain their historical C0."""
        value = self.extras.get("channel_index")
        if value is None:
            return 0 if self.modality == "dapi" else None
        return int(value)

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "modality": self.modality,
            "token": self.token,
            "path": str(self.path),
            "level": int(self.level),
        }
        if self.threshold is not None:
            out["threshold"] = int(self.threshold)
        if self.role == MOVING:
            out["orientation_case"] = int(self.orientation_case)
            if self.gui_affine is not None:
                out["gui_affine"] = self.gui_affine
        out.update(self.extras)
        return out


@dataclass(frozen=True)
class RunRoles:
    pair: Pair
    moving: ImageRole
    fixed: ImageRole

    @property
    def engine(self) -> str:
        return self.pair.engine

    @property
    def params_block(self) -> str:
        return self.pair.params_block

    @property
    def homography_stem(self) -> str:
        return self.pair.homography_stem

    def role(self, which: str) -> ImageRole:
        if which == MOVING:
            return self.moving
        if which == FIXED:
            return self.fixed
        raise ValueError(f"role must be one of {ROLES}, got {which!r}")


def normalize_pair_name(value: str) -> str:
    """Canonicalize a pair spelling, raising a ValueError that names the choices."""
    text = str(value).strip().lower().replace(" ", "")
    text = PAIR_ALIASES.get(text, text)
    if text not in PAIRS:
        raise ValueError(
            f"unknown pair {value!r}; choose from {', '.join(sorted(PAIRS))}"
        )
    return text


def pair_from_name(value: str) -> Pair:
    return PAIRS[normalize_pair_name(value)]


def pair_for(moving_modality: str, fixed_modality: str) -> Pair:
    """The pair two modalities name, or a ValueError listing what is expressible."""
    moving = str(moving_modality).strip().lower()
    fixed = str(fixed_modality).strip().lower()
    for pair in PAIRS.values():
        if pair.moving == moving and pair.fixed == fixed:
            return pair
    raise ValueError(
        f"no pair for moving={moving!r} fixed={fixed!r}; "
        f"defined pairs are {', '.join(sorted(PAIRS))}"
    )


def pair_from_mode(mode: str) -> Pair:
    """Read-only alias boundary for older commands; internal dispatch uses pairs."""
    text = str(mode).strip().lower()
    if text == "dapi":
        return PAIRS["dapi-he"]
    if text in ("he", "he0", "hefg"):
        return PAIRS["he-he"]
    raise ValueError(
        f"unknown mode {mode!r}; the pair names are {', '.join(sorted(PAIRS))}"
    )


def implemented_pairs() -> List[str]:
    return sorted(name for name, pair in PAIRS.items() if pair.implemented)


def coerce_pair(value) -> Pair:
    """Normalize public pair names; old mode tokens are accepted only as aliases."""
    if isinstance(value, Pair):
        return value
    if isinstance(value, RunRoles):
        return value.pair
    try:
        return pair_from_name(value)
    except ValueError:
        return pair_from_mode(value)


def add_pair_arguments(parser, *, default=None):
    """Canonical stage CLI interface with a hidden legacy --mode read boundary."""
    import argparse
    parser.add_argument("--pair", choices=implemented_pairs(), default=default)
    parser.add_argument("--mode", default=None, help=argparse.SUPPRESS)


def pair_from_args(args, *, run_dir=None):
    pair = coerce_pair(args.pair) if args.pair else None
    if args.mode:
        legacy = coerce_pair(args.mode)
        if pair is not None and pair != legacy:
            raise ValueError("--pair conflicts with legacy --mode")
        pair = legacy
    if run_dir is not None and (Path(run_dir) / "images_info.json").exists():
        recorded = read(run_dir).pair
        if pair is not None and pair != recorded:
            raise ValueError(f"requested {pair.name}, but this run records {recorded.name}")
        pair = recorded
    pair = pair or PAIRS[DEFAULT_PAIR]
    if not pair.implemented:
        raise ValueError(f"{pair.name} is not yet validated end to end")
    return pair


def artifact_path(run_dir, role, template):
    """Resolve a token-named artifact, allowing legacy he0/he files on old runs."""
    root = Path(run_dir)
    roles = read(root)
    tokens = [roles.role(role).token]
    if roles.pair.name == "he-he":
        tokens.append("he0" if role == MOVING else "he")
    candidates = [root / template.format(token=token) for token in tokens]
    return next((path for path in candidates if path.exists()), candidates[0])


def image_role(pair: Pair, role: str, path, level, *, threshold=None,
               orientation_case: int = 0, gui_affine=None, **extras) -> ImageRole:
    """Build one side of a binding, with its token derived from the pair."""
    return ImageRole(
        role=role,
        modality=pair.modality(role),
        path=str(path),
        level=int(level),
        token=pair.token(role),
        threshold=None if threshold is None else int(threshold),
        orientation_case=int(orientation_case),
        gui_affine=gui_affine,
        extras=dict(extras),
    )


def roles_block(pair: Pair, moving: ImageRole, fixed: ImageRole) -> Dict[str, Any]:
    """The `roles` block Stage 1 writes into `images_info.json`."""
    if moving.role != MOVING or fixed.role != FIXED:
        raise ValueError("roles_block takes one moving and one fixed ImageRole")
    if moving.modality != pair.moving or fixed.modality != pair.fixed:
        raise ValueError(
            f"{pair.name} is moving={pair.moving}/fixed={pair.fixed}, "
            f"got moving={moving.modality}/fixed={fixed.modality}"
        )
    return {
        "schema_version": 1,
        "pair": pair.name,
        MOVING: moving.as_dict(),
        FIXED: fixed.as_dict(),
    }


def flat_keys(pair: Pair, moving: ImageRole, fixed: ImageRole) -> Dict[str, Any]:
    """The per-side flat keys, spelled with each side's token.

    `dapi-he` preserves 4.10's keys -- `DAPI_path`, `HE_level`,
    `DAPI_orientation_case` and the rest -- so every stage that still reads them, and
    every run directory already on disk, is unaffected. New runs add optional channel
    keys. `he-he` produces `HEFG_*` and `HEBG_*` where 4.10 wrote `HE0_*` and `HE_*`.
    """
    out: Dict[str, Any] = {}
    for image in (moving, fixed):
        prefix = pair.key_prefix(image.role)
        out[f"{prefix}_path"] = str(image.path)
        out[f"{prefix}_level"] = int(image.level)
        for key in ("channel_index", "channel_name", "channel_source"):
            if key in image.extras:
                out[f"{prefix}_{key}"] = image.extras[key]
        if image.threshold is not None:
            out[f"{prefix}_threshold"] = int(image.threshold)
            if image.modality == "dapi":
                # Stage 2 and Stage 3 read the DAPI display cutoff under this name.
                out[f"{prefix}_LUT_threshold"] = int(image.threshold)
        if image.role == MOVING:
            out[f"{prefix}_orientation_case"] = int(image.orientation_case)
            if image.gui_affine is not None:
                out[f"{prefix}_gui_affine"] = image.gui_affine
    out["pair"] = pair.name
    return out


def _image_from_role_dict(role: str, pair: Pair, data: Dict[str, Any]) -> ImageRole:
    known = {"modality", "token", "path", "level", "threshold",
             "orientation_case", "gui_affine"}
    return ImageRole(
        role=role,
        modality=str(data["modality"]).strip().lower(),
        path=str(data["path"]),
        level=int(data["level"]),
        token=str(data.get("token") or pair.token(role)),
        threshold=None if data.get("threshold") is None else int(data["threshold"]),
        orientation_case=int(data.get("orientation_case", 0)),
        gui_affine=data.get("gui_affine"),
        extras={k: v for k, v in data.items() if k not in known},
    )


def _image_from_flat(info: Dict[str, Any], pair: Pair, role: str) -> ImageRole:
    prefix = pair.key_prefix(role)
    if pair.name == "he-he" and f"{prefix}_path" not in info:
        prefix = "HE0" if role == MOVING else "HE"
    if f"{prefix}_path" not in info:
        raise KeyError(
            f"images_info.json has no {role} image for pair {pair.name}: "
            f"expected a `{ROLES_KEY}` block or {prefix}_path"
        )
    threshold = info.get(f"{prefix}_threshold")
    if threshold is None:
        threshold = info.get(f"{prefix}_LUT_threshold")
    extras = {
        key: info[f"{prefix}_{key}"]
        for key in ("channel_index", "channel_name", "channel_source")
        if f"{prefix}_{key}" in info
    }
    return ImageRole(
        role=role,
        modality=pair.modality(role),
        path=str(info[f"{prefix}_path"]),
        level=int(info[f"{prefix}_level"]),
        token=pair.token(role),
        threshold=None if threshold is None else int(threshold),
        orientation_case=int(info.get(f"{prefix}_orientation_case", 0)),
        gui_affine=info.get(f"{prefix}_gui_affine"),
        extras=extras,
    )


def read(source) -> RunRoles:
    """Resolve a run's roles from a run dir, an images_info.json path, or its dict.

    Prefers the `roles` block, falls back to the flat keys. The fallback is what
    keeps every run directory written by 4.10 readable: a `dapi-he` run's flat keys
    are spelled the same in both versions.
    """
    if isinstance(source, dict):
        info = source
    else:
        path = Path(source)
        if path.is_dir():
            path = path / "images_info.json"
        info = json.loads(path.read_text(encoding="utf-8"))

    block = info.get(ROLES_KEY)
    if isinstance(block, dict) and block.get("pair"):
        pair = pair_from_name(block["pair"])
        return RunRoles(
            pair=pair,
            moving=_image_from_role_dict(MOVING, pair, block[MOVING]),
            fixed=_image_from_role_dict(FIXED, pair, block[FIXED]),
        )

    if info.get("pair"):
        pair = pair_from_name(info["pair"])
    elif info.get("mode") == "dapi" or "DAPI_path" in info:
        # 4.10 wrote `mode` only from the GUI, so the DAPI key is the other tell.
        pair = PAIRS["dapi-he"]
    elif "HEFG_path" in info or "HE0_path" in info:
        pair = PAIRS["he-he"]
    else:
        raise KeyError(
            f"images_info.json names no pair: no `{ROLES_KEY}` block, no `pair`, "
            f"and no DAPI_path or HEFG_path"
        )
    return RunRoles(
        pair=pair,
        moving=_image_from_flat(info, pair, MOVING),
        fixed=_image_from_flat(info, pair, FIXED),
    )


def describe(pair: Pair) -> str:
    return (f"{pair.name}: moving={pair.token(MOVING)} fixed={pair.token(FIXED)} "
            f"engine={pair.engine} parameters['{pair.params_block}']"
            + ("" if pair.implemented else "  [NOT IMPLEMENTED]"))
