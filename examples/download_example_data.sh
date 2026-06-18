#!/usr/bin/env bash
set -euo pipefail

# Download demo data for Xenium V1 Human Colon Cancer P1 CRC Add-on FFPE
# Outputs:
#   xenium_human_CRC_P1/he.ome.tif
#   xenium_human_CRC_P1/morphology_focus.ome.tif
#   xenium_human_CRC_P1/cells.csv.gz

OUTDIR="xenium_human_CRC_P1"
BASE_URL="https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE"

HE_FILE="Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE_he_image.ome.tif"
OUTS_ZIP_URL="${BASE_URL}/Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE_outs.zip"

mkdir -p "${OUTDIR}"
cd "${OUTDIR}"

echo "Downloading H&E image..."
curl -L -O "${BASE_URL}/${HE_FILE}"
mv -f "${HE_FILE}" "he.ome.tif"

echo "Downloading Xenium outs zip..."
curl -L -O "${OUTS_ZIP_URL}"

OUTS_ZIP_LOCAL=$(find . -maxdepth 1 -type f -name "*.zip" | head -n 1)

if [[ -z "${OUTS_ZIP_LOCAL}" ]]; then
    echo "Error: no .zip file found after download." >&2
    exit 1
fi

echo "Using outs zip: ${OUTS_ZIP_LOCAL}"

echo "Extracting morphology image..."
MORPH_IN_ZIP=$(unzip -Z1 "${OUTS_ZIP_LOCAL}" | grep -E '(^|/)morphology.*\.ome\.tif$' | head -n 1 || true)

if [[ -z "${MORPH_IN_ZIP}" ]]; then
    echo "Error: no morphology *.ome.tif found inside ${OUTS_ZIP_LOCAL}" >&2
    unzip -Z1 "${OUTS_ZIP_LOCAL}" | grep -E 'morphology|cells.csv' >&2 || true
    exit 1
fi

unzip -j -o "${OUTS_ZIP_LOCAL}" "${MORPH_IN_ZIP}"
mv -f "$(basename "${MORPH_IN_ZIP}")" "morphology_focus.ome.tif"

echo "Extracting cells.csv.gz..."
CELLS_IN_ZIP=$(unzip -Z1 "${OUTS_ZIP_LOCAL}" | grep -E '(^|/)cells\.csv\.gz$' | head -n 1 || true)

if [[ -z "${CELLS_IN_ZIP}" ]]; then
    echo "Error: cells.csv.gz not found inside ${OUTS_ZIP_LOCAL}" >&2
    unzip -Z1 "${OUTS_ZIP_LOCAL}" | grep -E 'cells' >&2 || true
    exit 1
fi

unzip -j -o "${OUTS_ZIP_LOCAL}" "${CELLS_IN_ZIP}"

echo "Cleaning up zip file..."
rm -f "${OUTS_ZIP_LOCAL}"

echo "Done. Files saved in ${OUTDIR}:"
ls -lh "he.ome.tif" "morphology_focus.ome.tif" "cells.csv.gz"