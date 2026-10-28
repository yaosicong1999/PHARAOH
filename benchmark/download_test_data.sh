#!/usr/bin/env bash
# Download the benchmark test dataset: 10x Genomics Xenium In Situ, human colorectal cancer
# Sample P1 (FFPE, Xenium v1 Human Colon panel with add-on), with its H&E image and the 8
# DAPI-H&E landmark pairs annotated by 10x Genomics.
#
#   ./download_test_data.sh
#
# Writes data/human_crc_p1/ (about 11.5 GB):
#   Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE_he_image.ome.tif            H&E, fixed image
#   Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE_he_imagealignment.csv       10x landmark pairs
#   morphology_focus/morphology_focus_000[0-3].ome.tif                          DAPI, moving image
# The DAPI image is extracted from the 13 GB Xenium output bundle, which is downloaded to the same
# folder and deleted afterwards (KEEP_ZIP=yes keeps it). Interrupted downloads resume when the
# script is run again; files already complete are skipped.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/data/human_crc_p1"
STEM="Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE"
BASE="https://cf.10xgenomics.com/samples/xenium/2.0.0/$STEM"
KEEP_ZIP="${KEEP_ZIP:-no}"

HE="${STEM}_he_image.ome.tif"
LANDMARKS="${STEM}_he_imagealignment.csv"
OUTS_ZIP="${STEM}_outs.zip"
# Sizes in bytes of the files as published, to catch a truncated download.
HE_BYTES=8664070297
LANDMARKS_BYTES=638
OUTS_ZIP_BYTES=13002851404
DAPI_FILES=(morphology_focus_0000.ome.tif morphology_focus_0001.ome.tif
            morphology_focus_0002.ome.tif morphology_focus_0003.ome.tif)

size_of() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1"; }

fetch() {  # fetch <file> <expected bytes>
    local name="$1" bytes="$2"
    if [ -f "$name" ] && [ "$(size_of "$name")" = "$bytes" ]; then
        echo "[download] $name already complete"
        return
    fi
    echo "[download] $name ($bytes bytes)"
    if ! { [ -f "$name.part" ] && [ "$(size_of "$name.part")" = "$bytes" ]; }; then
        curl -L --fail --retry 5 --retry-delay 10 -C - -o "$name.part" "$BASE/$name"
    fi
    if [ "$(size_of "$name.part")" != "$bytes" ]; then
        echo "[download] $name: expected $bytes bytes, got $(size_of "$name.part"); run the script again to resume" >&2
        exit 1
    fi
    mv "$name.part" "$name"
}

mkdir -p "$OUT"
cd "$OUT"

fetch "$LANDMARKS" "$LANDMARKS_BYTES"
fetch "$HE" "$HE_BYTES"

have_dapi=yes
for f in "${DAPI_FILES[@]}"; do [ -s "morphology_focus/$f" ] || have_dapi=no; done
if [ "$have_dapi" = yes ]; then
    echo "[download] morphology_focus/ already extracted"
else
    fetch "$OUTS_ZIP" "$OUTS_ZIP_BYTES"
    echo "[download] extracting morphology_focus/ from $OUTS_ZIP"
    unzip -o "$OUTS_ZIP" 'morphology_focus/*'
    for f in "${DAPI_FILES[@]}"; do
        [ -s "morphology_focus/$f" ] || { echo "[download] morphology_focus/$f missing from $OUTS_ZIP" >&2; exit 1; }
    done
fi
if [ "$KEEP_ZIP" != yes ] && [ -f "$OUTS_ZIP" ]; then
    rm -f "$OUTS_ZIP"
    echo "[download] removed $OUTS_ZIP (set KEEP_ZIP=yes to keep it)"
fi

echo "[download] done: $OUT"
ls -l "$OUT" "$OUT/morphology_focus"
