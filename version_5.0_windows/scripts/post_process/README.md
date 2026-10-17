# Apply a registration to images and coordinates

These tools read a finished PHARAOH run and apply its transformation. They never fit a new
registration and never modify their inputs. Run them from the version directory, in the
`PHARAOH` environment.

## Which transformation to pass

- `--run_dir RUN_DIR` selects the run's final transformation automatically: the canonical
  `<moving>_to_<fixed>_homography_level0.json`, which also carries a local correction field
  when the run accepted one, or the matrix-only `.csv` when the JSON is missing.
- `--matrix FILE` passes a transformation file directly: a PHARAOH transformation JSON, or a
  headerless 3 × 3 matrix as CSV or TXT. Prefer the JSON: for a run with a local correction,
  the CSV holds only the underlying homography.

The transformation maps moving (DAPI) level-0 pixels to fixed (H&E) level-0 pixels. Do not
pass 10x's own `*_he_imagealignment.csv` to these tools; it runs in the opposite direction (use
`xenium_align_10x_transcripts.py` for it).

## Units

Xenium tables store coordinates in **micrometers**. The tools divide them by the DAPI level-0
pixel size (`--dapi_mpp`, default 0.2125 µm/pixel, the Xenium morphology-image pixel size)
and then apply the transformation, so the output coordinates are **H&E level-0 pixels**. Use
`--dapi_mpp 1` when the input is already in DAPI level-0 pixels. Coordinate columns keep their
names but change units; every other column, every row and the row order are preserved. Output
tables are Parquet.

## Warp the whole moving image

```bash
python scripts/post_process/warp_dapi_to_he.py \
  --run_dir outputs/runs_YYYYMMDDHHMMSS \
  --output_img /path/to/dapi_in_he_space.ome.tif
```

The output is a tiled, pyramidal OME-TIFF on the H&E level-0 grid, with the H&E pixel size.
With `--run_dir`, the input images default to the paths recorded in the run; `--dapi_img` and
`--he_img` override them. With `--matrix`, pass `--dapi_img` and either `--he_img` or
`--output_size`. The output must end in `.ome.tif`. Other options: `--channel` (default 0),
`--interpolation` (`linear` by default; `nearest`, `cubic`, `lanczos`), `--output_size
WIDTHxHEIGHT`, `--output_mpp`, `--compression` and `--no_pyramid`. Memory stays bounded
because the image is warped block by block.

## Xenium transcripts, cells and boundaries

Transcripts (`transcripts.parquet`, `.csv.gz`, `.csv` or the run's `transcripts.zarr.zip`):

```bash
python scripts/post_process/xenium_align_transcripts.py \
  --run_dir outputs/runs_YYYYMMDDHHMMSS \
  --input_path /path/to/transcripts.parquet \
  --output_path /path/to/transcripts_he.parquet
```

Cell centroids, cell boundaries and nucleus boundaries:

```bash
for table in cells cell_boundaries nucleus_boundaries; do
  python "scripts/post_process/xenium_align_${table}.py" \
    --matrix outputs/runs_YYYYMMDDHHMMSS/dapi_to_he_homography_level0.json \
    --input_path "/path/to/xenium_outs/${table}.parquet" \
    --output_path "/path/to/aligned/${table}_he.parquet"
done
```

| tool | input table | coordinate columns |
| --- | --- | --- |
| `xenium_align_transcripts.py` | `transcripts.parquet` / `.csv.gz` / `.zarr.zip` | `x_location`, `y_location` |
| `xenium_align_cells.py` | `cells.parquet` / `cells.csv.gz` | `x_centroid`, `y_centroid` |
| `xenium_align_cell_boundaries.py` | `cell_boundaries.parquet` / `.csv.gz` | `vertex_x`, `vertex_y` |
| `xenium_align_nucleus_boundaries.py` | `nucleus_boundaries.parquet` / `.csv.gz` | `vertex_x`, `vertex_y` |

`--x_column` and `--y_column` choose other columns, and `--chunk_size` (default 300,000 rows)
limits memory. The boundary tools transform every vertex and keep `cell_id`, `label_id` and
the vertex order unchanged. Areas such as `cell_area` stay in µm².

**10x's own alignment.** `xenium_align_10x_transcripts.py` does the same with the
`*_he_imagealignment.csv` that 10x provides with some H&E images. That matrix maps H&E pixels to
DAPI pixels, so the tool inverts it before use (`--assume-direction` if a dataset uses the other
convention).

## CosMx cell boundaries

```bash
python scripts/post_process/cosmx_align_cell_boundaries.py \
  --run_dir outputs/runs_YYYYMMDDHHMMSS \
  --input_path /path/to/S0-polygons.csv \
  --metadata /path/to/S0_metadata_file.csv \
  --output_path /path/to/cell_boundaries_he.parquet
```

For a DAPI image stitched from the CosMx FOVs at their global coordinates. Polygon vertices are
first placed on that canvas (`x - gx_min`, `gy_max - y`), with the canvas origin taken from the
metadata file, from `--gx_min` and `--gy_max`, or inferred from the polygon export itself, and
then transformed. Every input column is kept and `x_he_level0_px`, `y_he_level0_px` are
appended.

## Bin transcripts and plot genes

Bin aligned transcripts (H&E level-0 pixels) into 8-µm bins:

```bash
python scripts/post_process/xenium_bin_transcripts.py \
  --input_path /path/to/transcripts_he.parquet \
  --output_dir /path/to/bins_8um \
  --raw_mpp 0.2738 --target_mpp 0.5 --bin_size_um 8
```

Plot genes, or the total counts, on the H&E image:

```bash
python scripts/post_process/xenium_plot_binned_gene.py \
  --input /path/to/bins_8um/cnts.parquet --genename EPCAM PTPRC \
  --he_img /path/to/he.ome.tif --output_dir /path/to/plots
python scripts/post_process/xenium_plot_binned_umi.py \
  --input /path/to/bins_8um/cnts.parquet --he_img /path/to/he.ome.tif \
  --output_img /path/to/plots/umi.png
```

`--raw_mpp` is the H&E level-0 pixel size (print it with
`python scripts/post_process/get_he_raw_mpp.py --he_img /path/to/he.ome.tif`), and
`--target_mpp` the resolution of the bin grid. The bin size is given in micrometers
(`--bin_size_um`) or in target pixels (`--bin_size_target_px`). The output is a bin-by-gene
`cnts.parquet` with the bin positions in `locs.tsv`.

## Check a registration visually

```bash
python scripts/post_process/xenium_plot_nucleus_rois.py \
  --dapi /path/to/morphology_focus.ome.tif \
  --he /path/to/he.ome.tif \
  --matrix outputs/runs_YYYYMMDDHHMMSS/dapi_to_he_homography_level0.json \
  --nucleus-boundaries /path/to/nucleus_boundaries.parquet \
  --n-points 9 --tile-size-um 100 \
  --output-path /path/to/nucleus_roi_qc.png
```

One PNG with one row per region: the DAPI image with the raw Xenium nucleus outlines, and the H&E
image with the same outlines placed by the registration. Regions are spread over the tissue by
seeded farthest-point sampling (`--seed`, default 42), not chosen for good alignment, and
`--tile-size-um` is the full side of each square. The DAPI pixel size is read from the image;
`--dapi-mpp` supplies it when missing. Pass the **raw** nucleus boundaries, not an aligned table.

## Other utilities

- `convert_to_rescaled_image.py`: write an OME-TIFF as JPG, PNG or TIFF at a chosen pixel size
  (`--input_img`, `--output_img`, `--output_mpp`).
- `get_he_raw_mpp.py`: print the level-0 pixel size of an OME-TIFF.

`pharaoh_transform.py` is the shared module that reads a transformation and applies it; it is
not run on its own.
