# Example data

This directory contains scripts for obtaining example datasets (Xenium Human Colon Cancer P1) used to demonstrate the PHARAOH workflow.


### Download the example dataset

```bash
bash download_example_data.sh
```

This script downloads a publicly available Xenium Human Colon Cancer P1 dataset from 10x Genomics and prepares the required H&E image, morphology image, and optional cell information for PHARAOH.
### Additional datasets

Additional datasets used in this study can be obtained using the links and accession numbers provided in the Data Availability section of the manuscript.

The `download_data.sh` script serves as an example for downloading and preparing Xenium datasets for use with PHARAOH. The download URLs and filename patterns can be modified for other Xenium datasets listed in the Data Availability section. Note that file and directory names may vary across Xenium platform versions and dataset releases.

### Running PHARAOH

For installation instructions, software requirements, and workflow examples, please refer to the main repository README.

### Provided alignment files

In addition to the data-download script, this directory includes example alignment JSON files for reviewer testing.

- Manual initial-alignment JSON files can be loaded directly during the manual-alignment step.
- Final alignment-result JSON files are provided for the Xenium testing datasets used in the manuscript.

These files allow users to reproduce or inspect downstream PHARAOH steps without repeating the full manual initialization procedure.
