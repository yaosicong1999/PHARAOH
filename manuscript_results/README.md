# PHARAOH results of the manuscript

`pharaoh/` holds the final transformation of every PHARAOH registration reported in the
manuscript: 123 files, one per registration, grouped by experiment. Each is the run's own
`<moving>_to_<fixed>_homography_level0.json`, copied unchanged and prefixed with the run's name.
`pharaoh/index.csv` lists every file with its dataset, condition, moving and fixed side,
transformation type and number of inliers.

## What a file contains

Every file maps **moving level-0 pixel coordinates to fixed level-0 pixel coordinates** of the
original images. The 3 × 3 matrix is under `homography_3x3`; `transform_type` says whether
PHARAOH selected an `affine` or a `homography` model (an affine matrix is stored in the same
3 × 3 form). Three runs (`adjacent_he_he`, `codex`, `cycif`) are `local_residual`: their final
mapping is the matrix under `base_homography_3x3` plus the smooth local correction field
stored under `local_residual`. The remaining keys record how the transformation was fitted
(correspondence counts, inliers, residuals and the model-selection details).

To apply a file to an image or to coordinates, use the tools in
`version_5.0_macos_linux/scripts/post_process/` (pass the file with `--matrix`); they handle all
three transformation types.

## Experiments

| folder | registrations | pair (moving → fixed) | experiment |
| --- | --- | --- | --- |
| `xenium_keypoints` | 6 | DAPI → H&E | benchmark, datasets with DAPI–H&E landmarks |
| `xenium_matrix` | 19 | DAPI → H&E | benchmark, datasets with a 10x alignment matrix |
| `xenium_experimental` | 30 | DAPI → H&E | Xenium sections without a reference alignment (visual assessment) |
| `cosmx_subsampled` | 21 | DAPI → H&E | CosMx kidney slide: all 483 FOVs, and 20%, 40%, 60% and 80% of them with five random seeds |
| `orion_perturbed` | 20 | Hoechst → H&E | four Orion images, each with five seeded synthetic distortions of the Hoechst channel |
| `hyreco_phh3` | 6 | PHH3 hematoxylin → H&E | restained PHH3–H&E pairs from HyReCo |
| `simulated_dapi_dapi` | 9 | DAPI → DAPI | partially overlapping same-section crops, three datasets × three perturbations |
| `simulated_he_he` | 9 | H&E → H&E | the same crops of the H&E images |
| `adjacent_he_he` | 1 | H&E → H&E | Visium HD H&E onto the adjacent Xenium H&E section (colorectal cancer P2) |
| `codex` | 1 | CODEX DNA → H&E | CODEX image onto an adjacent H&E section |
| `cycif` | 1 | CyCIF DNA → H&E | CyCIF image onto an adjacent H&E section |

## Notes

- **Benchmark series.** The benchmark was run three times with identical inputs and settings;
  accuracy is reported from series 1, and these are the series-1 transformations.
- **Mirrored H&E.** For `human_breast_cancer_s1r1` and `human_breast_cancer_s1r2` the released
  H&E is mirrored relative to the DAPI. In the benchmark, every method received the H&E flipped
  vertically (y → H − 1 − y), so these two files map DAPI pixels to the **flipped** H&E.
- **CosMx.** The moving image is the DAPI canvas stitched from the FOVs at their global
  coordinates; file names give the seed and the share of FOVs kept (`cosmx_100percent` is the
  reference registration with every FOV).
- **Orion.** The moving image is the Hoechst channel after a known synthetic distortion, so each
  transformation can be compared with the inverse of that distortion.
- **HyReCo.** Cases 464 and 611 were registered but not evaluated, because their landmark
  annotations were too imprecise.
- **Parameters.** All runs used the default parameters with two exceptions: the Stage-4 tile
  fraction was 1.0 (every Stage-3 tile) in all experiments other than the Xenium benchmark and
  the experimental Xenium sections, which used the default 0.30; and the CODEX and CyCIF runs
  used nuclear-density anchors (`stage4b.anchor_mode = density`) instead of the automatic choice.

Dataset sources are listed in Supplementary Table 1, and the experiments are described in the
Supplementary Methods.
