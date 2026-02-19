## HistoPyrAlign (Beta version 2.0)

HistoPyrAlign is a scalable and generalizable framework for multimodal tissue image alignment and spatial transcriptomics enhancement.

It enables fast, robust, GUI-based, and GPU-free semi-automatic registration between DAPI imaging from multiplexed imaging platforms (Xenium, CosMx, CODEX, CyCIF), and histology (H&E), supporting both same-section and adjacent-section alignment.



## 📂 Required Input Files

In order to run the HistoPyrAlign platform, you will need the following files:

1. An image containing one DAPI channel (should be on the first channel if of multi-channels), in the format of `.ome.tif`, `.tif` or `.jpg`. 

    The DAPI file from Xenium platfrom typically is in the format of either `morphology_focus.ome.tif` or `morphology_focus/morphology_focus_0000.ome.tif`
2. An H&E image from the same slice or an adjacent slice in the format of `.ome.tif`, `.tif` or `.jpg`. 
3. (Optional) A `cells.csv.gz` containing cell centroids information for visualization purpose for Xenium platform.

In order to achieve the best alignment performance, please consider using raw, unscaled images.

---

## ⚙️ Setup

First, clone the repository:
```bash
git clone https://github.com/yaosicong1999/HistoPyrAlign.git
cd HistoPyrAlign/version_2.0
```
We recommend installing Pixel2Gene in a dedicated Conda environment. 
```bash
bash install_conda_env.sh
```
Typical installation time is approximately 10–20 minutes on a standard Linux desktop with internet access, depending on network speed and package resolution.

### Tested environments
HistoPyrAlign has been tested on MacOS 15.7.3.

---


## 🚀 Usage

For parameter controls, please see the later subsection. 
### Overall control panel

To launch the overall control panel, just:
```bash
python 0_pipeline.py
```

In order to create a new run attempt, click the `New RUN_DIR` button on the top-right corner. This will create a run folder in the format of `/Current folder/runs_YYYYMMDDHHMMSS/`.

If you want to load a previous existing run attempt, please click `Choose RUN_DIR` button. Please note that the run folder should be in the format of  `/runs_{some_integer_ID}/`.


Please note that the following steps may need a long time (~1 to 2 mins) to load dependecies for the first time use after opening the control panel for the first time.

### Stage 1: Select H&E image and DAPI image
Simply click the `Stage 1` button in the control panel.

Then, in the viewer:

- Click `Select H&E Image`
- Click `Select DAPI Image`

Large `.tif` or `.jpg` files may take longer to load due to reading and downsampling. `.ome.tif` files typically load within a few seconds.

#### Adjust Visualization

After loading both images, use the two `threshold sliders` to adjust:
  - The *H channel* visualization for the H&E image (displayed in the second row)
  - The *LUT-colored* DAPI image (displayed in the second row)

For the DAPI image:

> Ensure the LUT-colored image is clearly visible but not overly saturated or patchy.  
> Proper visualization at this stage will make subsequent alignment steps easier and more reliable.

#### ⚠️ Required: Match Orientation

Before proceeding:

- Use the `Rotate` and `Flip` buttons in the DAPI column  
- Match the DAPI image orientation to the H&E image

This step is mandatory.


#### Save Orientation

Once everything looks correct, click `Confirm & Save Orientation`.

#### Output

After completing Stage 1, the following outputs will be generated:

> - `images_info.json`
> - PNG images prefixed with `1_`

---

### Stage 2: Get Initial Alignment

Simply click the `Run Stage 2B: Manual Alignment` button in the control panel.  
(`Run Stage 2A: Blob Matching` is currently suspended.)

#### Alignment Modes

By default, the alignment mode is set to `Mode: Affine`

Under `Mode: Affine`, you can:

- Drag any blue corner to scale the floating DAPI image.
- Hold the `Shift` key while dragging a blue corner to scale the image proportionally (diagonal scaling).
- Drag the floating DAPI image to move it.
- Hold / release the `Control` key to hide or reveal the floating DAPI image.

Once the approximate size and position are matched, click the  
`Mode: Affine` button to switch to `Mode: Perspective`.

Under `Perspective mode`, you can:

- Drag any blue corner to stretch or distort the floating DAPI image.

> ⚠️ Rotation has not been implemented as a standalone function yet.  
> To simulate rotation, use `Perspective mode` adjustments.

#### Loading Existing Alignment

To load a previously saved manual alignment:

1. Click `Load H (.json)`.
2. Select the transformation matrix file.

> ⚠️ The transformation must correspond to the same pyramid level specified in `images_info.json` from Stage 1.

#### Saving Alignment

Once alignment is satisfactory, click `Save Alignment` at the bottom of the viewer.

#### Output

After completing Stage 2, the following outputs will be generated:

> - `manual_initial_alignment.json`
> - PNG images prefixed with `2_`

---

###  Stage 3: Extract Tiles

Simply click the `Stage 3` button in the control panel to open the tile gallery.

#### Available Controls in the Step 3 Viewer
There are three buttons in the Step 3 viewer:

1. `Sample Tile Centroids`  
   Samples tile centroids based on the parameters defined in `parameters.json`.  
   If the requested number of tiles or tile size would result in oversampling of the image space, the algorithm will automatically reduce the number of sampled tiles to avoid excessive overlap.

2. `Tile Pilot Examination`  
   Extracts 10 pilot tiles from the sampled tiles for quick parameter tuning.

   You can adjust:
   - `DAPI masking offset`  
     - Positive values → smaller nuclei regions  
     - Negative values → larger nuclei regions  
   - `H&E intensity threshold (range: 0–1)`  
     - Higher values → larger nuclei regions  
     - Lower values → smaller nuclei regions  

   These parameters will be saved and applied in the main nuclei masking step.

   ⚠️ This step is optional, but strongly recommended.  
   It improves robustness of downstream alignment, especially for:
   - Low-quality Xenium data  
   - Adjacent tissue slices  

3.` Extract Current Tiles`  
   Extracts all sampled tiles based on the sampled centroid locations.

#### Output

After completing Stage 3, the following outputs will be generated:

> - `sampled_points.json`
> - `tiles/` directory
> - `pilot_tiles/` directory (if pilot examination was performed)
> - PNG images prefixed with `3_`

---

### Stage 4: Extract Nuclei Patches

Simply click the `Stage 4` button in the control panel to open the tile gallery.

#### Available Controls in the Step 4 Viewer
There are six buttons in the Stage 4 viewer:

1. `Previous / Next / Refresh`  
   Navigate to the previous or next tile, or refresh the tiles and masks in the gallery.

2. `Run Nuclei Masking`   
   Generates nuclei masks for each DAPI tile and each H&E tile.

3. `Run Standout Nuclei Detection`  
   Aligns the DAPI nuclei mask and H&E nuclei mask for each tile pair (after masking is completed for all tiles).  

   - For high-quality Xenium data, this step typically identifies multiple standout nuclei as anchor points.  
   - For lower-quality Xenium data or adjacent tissue slices, it may not detect enough standout nuclei. In such cases, if the mask pair still yields a reasonable global alignment, the aligned tile centers will be used as anchor points instead.  

   This step usually takes approximately 3–5 minutes.

4. `Run Nuclei Patch Cropping`  
   Extracts paired DAPI and H&E patches based on the detected standout nuclei or aligned centers.

#### Output

Stage 4 generates an output folder `nuclei patches`.

---

###  Stage 5: View Nuclei Patches and Get Final Alignment
Click the `Stage 5` button in the control panel to open the nuclei patch gallery.

This viewer displays the extracted nuclei patches (or patches centered at the aligned centroids) for both DAPI and H&E images, allowing visual inspection. You can click any image to enlarge it.

> ⚠️ **Note:** This feature is still under development.  
> In the enlarged view, you can click on the image to propose refined keypoints.  
> Press `Enter` to save the selected point.
> 
#### Available Controls in the Step 5 Viewer
There are five buttons in the viewer:

1. `Previous / Next`  
Navigate between nuclei pairs.
2. `Calculate alignment matrix`  
Computes the final alignment matrix using the currently available keypoints (centroids).
The result is saved as: `dapi_to_he_homography_level0.json`
3. `Hide / Unhide auto centroids`   
Toggle the visibility of automatically detected centroids in each image.
4. `DAPI: LUT / Raw`
Switch between LUT-colored DAPI visualization and DAPI intensity image.

#### Output
After clicking `Calculate alignment matrix`, Stage 5 generates `dapi_to_he_homography_level0.json`.  
This file contains the final homography matrix mapping DAPI (level 0) coordinates to H&E (level 0) coordinates.

---

### Stage 6: View Final Alignment

Click the **Stage 6** button in the control panel to open the final alignment gallery.

#### Available Controls in the Step 6 Viewer

There are three buttons in the viewer:

1. `Load Keypoints + Alignment Matrix`  
   Loads the keypoints onto both the DAPI and H&E images and generates the overlay visualization.

2. `Toggle H&E / Overlay `   
   Switches between:
   - H&E image only  
   - H&E image with DAPI overlay  

3. `Load Cell Data (cells.csv.gz)`  
   Loads cell data (for the Xenium platform) and overlays cell centroids on the H&E image for visualization.

#### Output

After completing Stage 6, the following outputs will be generated:

- PNG images prefixed with `9_`
- GIF images prefixed with `9_`

---