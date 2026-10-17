#!/usr/bin/env bash
set -euo pipefail

# Download demo data for the H&E-FG -> H&E (he-he) alignment mode.
#
# Moving  (H&E-FG): Visium HD Human Colon Cancer P2 tissue image (the H&E of the
#                   adjacent-section H&E-H&E experiment in the manuscript)
# Fixed   (H&E)   : Xenium V1 Human Colon Cancer P2 CRC Add-on FFPE H&E image
#
# The Visium HD H&E (adjacent section) is warped/placed onto the
# fixed Xenium H&E. The Visium HD .btf has no pyramid, so convert it once to a
# pyramidal OME-TIFF, then run from version_5.0_macos_linux/:
#   python scripts/pre_process/convert_image_to_ome_tif.py ../examples/he_he_human_CRC_P2/hefg.btf
#   ./run_pharaoh_cli.sh --pair he-he \
#                        --moving "../examples/he_he_human_CRC_P2/hefg.ome.tif" \
#                        --fixed  "../examples/he_he_human_CRC_P2/he.ome.tif"
#
# Outputs:
#   he_he_human_CRC_P2/hefg.btf     (moving H&E-FG, Visium HD)
#   he_he_human_CRC_P2/he.ome.tif   (fixed  H&E,    Xenium)

OUTDIR="he_he_human_CRC_P2"

HEFG_URL="https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Human_Colon_Cancer_P2/Visium_HD_Human_Colon_Cancer_P2_tissue_image.btf"
HE_URL="https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_he_image.ome.tif"

HEFG_FILE="$(basename "${HEFG_URL}")"
HE_FILE="$(basename "${HE_URL}")"

mkdir -p "${OUTDIR}"
cd "${OUTDIR}"

echo "Downloading moving H&E-FG image (Visium HD tissue image)..."
curl -L -O "${HEFG_URL}"
mv -f "${HEFG_FILE}" "hefg.btf"

echo "Downloading fixed H&E image (Xenium)..."
curl -L -O "${HE_URL}"
mv -f "${HE_FILE}" "he.ome.tif"

echo "Done. Files saved in ${OUTDIR}:"
ls -lh "hefg.btf" "he.ome.tif"
