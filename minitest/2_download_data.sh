#!/usr/bin/env bash
# Step 2: the two minitest datasets from 10x Genomics, into data/.
#
#   ./2_download_data.sh
#
#   data/human_crc_p1/   landmark dataset, about 11.5 GB kept (22 GB downloaded): H&E, the 8 10x
#                        landmark pairs, and morphology_focus/ (DAPI) from the 13 GB output bundle
#   data/human_heart/    matrix dataset, about 0.9 GB kept (4 GB downloaded): H&E, 10x's
#                        DAPI-to-H&E matrix, and morphology_focus.ome.tif (DAPI) and
#                        transcripts.parquet from the 3.5 GB output bundle
#
# Each file's size is checked against the published one. Interrupted downloads resume when the
# script is run again; complete files are skipped. The output bundles are deleted once their
# files are extracted (KEEP_ZIP=yes keeps them). If ../benchmark/data/human_crc_p1 is already
# complete (../benchmark/download_test_data.sh), data/human_crc_p1 links to it instead.
set -euo pipefail
source "$(dirname "$0")/settings.sh"
KEEP_ZIP="${KEEP_ZIP:-no}"

size_of() { stat -L -f %z "$1" 2>/dev/null || stat -L -c %s "$1"; }   # -L: a linked file counts

fetch() {  # fetch <base url> <file> <expected bytes>, into the current folder
    local base="$1" name="$2" bytes="$3"
    if [ -f "$name" ] && [ "$(size_of "$name")" = "$bytes" ]; then
        echo "[download] $name already complete"
        return
    fi
    echo "[download] $name ($bytes bytes)"
    if ! { [ -f "$name.part" ] && [ "$(size_of "$name.part")" = "$bytes" ]; }; then
        curl -L --fail --retry 5 --retry-delay 10 -C - -o "$name.part" "$base/$name"
    fi
    if [ "$(size_of "$name.part")" != "$bytes" ]; then
        echo "[download] $name: expected $bytes bytes, got $(size_of "$name.part"); run the script again to resume" >&2
        exit 1
    fi
    mv "$name.part" "$name"
}

extracted() {  # extracted <file> <expected bytes>: is it there, complete?
    [ -f "$1" ] && [ "$(size_of "$1")" = "$2" ]
}

drop_zip() {
    if [ "$KEEP_ZIP" != yes ] && [ -f "$1" ]; then
        rm -f "$1"
        echo "[download] removed $1 (set KEEP_ZIP=yes to keep it)"
    fi
}

# ---- human colorectal cancer P1: 8 landmark pairs ------------------------------------------------
CRC_STEM="Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE"
CRC_BASE="https://cf.10xgenomics.com/samples/xenium/2.0.0/$CRC_STEM"
CRC_FILES=("${CRC_STEM}_he_imagealignment.csv:638" "${CRC_STEM}_he_image.ome.tif:8664070297")
CRC_ZIP="${CRC_STEM}_outs.zip"; CRC_ZIP_BYTES=13002851404
CRC_DAPI=(morphology_focus_0000.ome.tif morphology_focus_0001.ome.tif
          morphology_focus_0002.ome.tif morphology_focus_0003.ome.tif)

crc_complete() {  # crc_complete <folder>
    local spec f
    for spec in "${CRC_FILES[@]}"; do extracted "$1/${spec%%:*}" "${spec##*:}" || return 1; done
    for f in "${CRC_DAPI[@]}"; do [ -s "$1/morphology_focus/$f" ] || return 1; done
}

CRC_OUT="$MINITEST_DIR/data/human_crc_p1"
mkdir -p "$MINITEST_DIR/data"
if [ ! -e "$CRC_OUT" ] && crc_complete "$BENCH_DIR/data/human_crc_p1"; then
    ln -s "../../benchmark/data/human_crc_p1" "$CRC_OUT"
    echo "[download] human_crc_p1: linked to the complete copy in ../benchmark/data/"
fi
if crc_complete "$CRC_OUT"; then
    echo "[download] human_crc_p1 already complete"
else
    mkdir -p "$CRC_OUT" && cd "$CRC_OUT"
    for spec in "${CRC_FILES[@]}"; do fetch "$CRC_BASE" "${spec%%:*}" "${spec##*:}"; done
    fetch "$CRC_BASE" "$CRC_ZIP" "$CRC_ZIP_BYTES"
    echo "[download] extracting morphology_focus/ from $CRC_ZIP"
    unzip -o -q "$CRC_ZIP" 'morphology_focus/*'
    crc_complete "$CRC_OUT" || { echo "[download] morphology_focus/ incomplete after extraction" >&2; exit 1; }
    drop_zip "$CRC_ZIP"
fi

# ---- human heart: every transcript against 10x's matrix ------------------------------------------
HEART_STEM="Xenium_V1_hHeart_nondiseased_section_FFPE"
HEART_BASE="https://cf.10xgenomics.com/samples/xenium/1.9.0/$HEART_STEM"
HEART_FILES=("${HEART_STEM}_he_imagealignment.csv:126" "${HEART_STEM}_he_image.ome.tif:540397952")
HEART_ZIP="${HEART_STEM}_outs.zip"; HEART_ZIP_BYTES=3456638778
HEART_EXTRACT=("morphology_focus.ome.tif:274743084" "transcripts.parquet:44446150")

HEART_OUT="$MINITEST_DIR/data/human_heart"
mkdir -p "$HEART_OUT" && cd "$HEART_OUT"
for spec in "${HEART_FILES[@]}"; do fetch "$HEART_BASE" "${spec%%:*}" "${spec##*:}"; done
missing=()
for spec in "${HEART_EXTRACT[@]}"; do extracted "${spec%%:*}" "${spec##*:}" || missing+=("${spec%%:*}"); done
if [ ${#missing[@]} -eq 0 ]; then
    echo "[download] morphology_focus.ome.tif and transcripts.parquet already extracted"
else
    fetch "$HEART_BASE" "$HEART_ZIP" "$HEART_ZIP_BYTES"
    echo "[download] extracting ${missing[*]} from $HEART_ZIP"
    unzip -o -q "$HEART_ZIP" "${missing[@]}"
    for spec in "${HEART_EXTRACT[@]}"; do
        extracted "${spec%%:*}" "${spec##*:}" || { echo "[download] ${spec%%:*} incomplete after extraction" >&2; exit 1; }
    done
fi
drop_zip "$HEART_ZIP"

_log "[download] done:"
du -shL "$MINITEST_DIR/data/human_crc_p1/" "$MINITEST_DIR/data/human_heart/"
_log "[download] next: ./3_run.sh"
