"""Role-token pilot overrides shared by tile and nucleus-patch masking."""
import math


def apply_token_overrides(params, preview):
    overrides = dict(params.get("_token_overrides", {}))
    # Old pilot files tuned modalities globally. Read those as low-priority
    # side overrides too, so raw per-token parameters cannot hide saved choices.
    for token in ("dapi", "he", "hefg", "hebg", "dapifg", "dapibg"):
        legacy_key = "dapi_thr_offset" if token.startswith("dapi") else "he_intensity_threshold"
        key = "thr_offset" if token.startswith("dapi") else "mask_intensity_threshold"
        if legacy_key in preview:
            modality = "dapi" if token.startswith("dapi") else "he"
            overrides[f"{token}_{key}"] = params[f"{modality}_{key}"]
    for token, values in preview.get("sides", {}).items():
        if token not in ("dapi", "he", "hefg", "hebg", "dapifg", "dapibg"):
            raise ValueError(f"Unknown pilot side token: {token}")
        key = "thr_offset" if token.startswith("dapi") else "mask_intensity_threshold"
        if key not in values:
            continue
        value = float(values[key])
        lo, hi = (-100, 100) if key == "thr_offset" else (0, 1)
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"{token}.{key} must be in [{lo}, {hi}]")
        overrides[f"{token}_{key}"] = int(value) if key == "thr_offset" else value
    params["_token_overrides"] = overrides
    return params


def side_value(params, token, modality, key):
    name = f"{token}_{key}"
    value = params.get(f"{modality}_{key}")
    # Modality defaults have already been parsed/validated by the stage loader.
    # Consult raw keys only for a genuinely distinct same-modality side token.
    if token != modality:
        value = params.get("_raw_block", {}).get(name, value)
    return params.get("_token_overrides", {}).get(name, value)
