# Scripts

Start runs from the launchers in the version directory (`run_pharaoh_cli`, `run_pharaoh_parallel`,
`run_pharaoh_failsafe`, `run_pharaoh_gui`). This directory holds the pipeline itself and a few
tools you may run directly.

## Files you may edit or run

| path | use |
| --- | --- |
| `parameters.json` | every pipeline parameter: one block per pair (`dapi-he`, `he-dapi`, `he-he`, `dapi-dapi`), one block per stage inside it, and the default quality-control image groups (`qc_images`) |
| `pre_process/convert_image_to_ome_tif.py` | convert a `.btf`, `.tif`, `.ndpi`, `.svs`, `.jpg` or `.czi` image to a pyramidal OME-TIFF (the launchers do this automatically for non-OME-TIFF inputs) |
| `post_process/` | apply a finished registration to the moving image, Xenium transcripts, cells and boundaries, or CosMx polygons; bin and plot transcripts; check a registration visually. See [`post_process/README.md`](post_process/README.md) |
| `render_dapi_preview.py` | redraw `1_dapi_preview.png` for an existing run directory |
| `batch_timeline.py` | redraw `batch_timeline.html` for a finished batch: `python scripts/batch_timeline.py <output-root>` |

A parameter change applies to new runs. Resuming a run directory whose parameter block has
changed is refused until `--force` is passed, so a run's outputs always match the
`run_parameters.json` it records.

## DAPI channel selection

For every DAPI side, Stage 1 reads the OME channel names and takes, in order, DAPI, Hoechst,
`DNA`, `DNA (0)`, then the first other DNA-like name; if none matches, it warns and uses channel
0. `--moving-channel C` or `--fixed-channel C` (zero-based) chooses the channel for a DAPI side
of `--pair`. These select a channel; `--stage1-moving-level` and `--stage1-fixed-level` select a
pyramid level.

## Pipeline code

The remaining files are the pipeline and are not run directly: `cli_stage*.py` (one per
headless stage), `engine_dapi/` (the stage implementations, used for all four pairs),
`gui_*.py`, `gui_fail_safe/` and `gui_pilot/` (the interactive windows), `scheduling.py` and
`monitoring.py` (batch admission control and memory monitoring), and shared modules such as
`my_utils.py`, `run_roles.py` and `stage5_fit.py`.
