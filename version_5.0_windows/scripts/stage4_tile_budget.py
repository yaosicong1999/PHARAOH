"""How many Stage-3 tiles Stage 4 works on: one rule for Stage 4a's masks and Stage 4b's scoring.

Stage 3 sizes its own tile count from the tissue it measured, so the Stage-4 count
follows as `stage3.stage4_tile_fraction` of what Stage 3 actually produced, unless
`stage4b.n_tiles` names a number outright. Stage 4a persists that many masks and
Stage 4b scores them; both call this so neither can carry a default of its own
(Stage 4b used to say 60 while Stage 4a masked 30%, and the 60 only ever applied when
more tiles were masked than the rule asked for).
"""


def stage4b_tile_count(available_tiles=None, *, pair=None, default=60):
    """Tiles Stage 4 works on, from `available_tiles` Stage-3 tile pairs."""
    from my_utils import cli_params_block

    params = cli_params_block(pair)
    stage4b = params.get("stage4b", {})
    explicit = stage4b.get("n_tiles", stage4b.get("n_tiles_for_nuclei_detection"))
    if explicit is not None:
        return max(1, int(explicit))
    if available_tiles:
        fraction = float(params.get("stage3", {}).get("stage4_tile_fraction", 0.30))
        return max(1, int(round(int(available_tiles) * fraction)))
    return max(1, int(default))
