#!/usr/bin/env bash
set -e

ENV_NAME="PHARAOH"
PYTHON_VERSION="3.10"

echo "Creating conda environment: ${ENV_NAME}"
conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION}

echo "Activating environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "Upgrading pip"
python -m pip install --upgrade pip setuptools wheel

echo "Installing packages (this may take a while)..."

pip install \
numpy==2.2.6 \
pandas==2.3.3 \
scipy==1.15.3 \
matplotlib==3.10.0 \
pillow==11.3.0 \
opencv-python==4.12.0.88 \
tifffile==2025.5.10 \
imagecodecs==2024.9.22 \
ome-types==0.6.3 \
pyvips==3.1.1 \
shapely==2.1.1 \
scikit-image==0.25.2 \
scikit-learn==1.7.2 \
tqdm==4.67.1 \
psutil==7.2.2 \
PyQt5==5.15.11 \
zarr \
pyarrow \
aicspylibczi==4.0.0 \

echo "Verifying OpenCV import"
python -c "import cv2; print('cv2 OK:', cv2.__version__)"

# scripts/pre_process converts .czi input through libCZI.
echo "Verifying CZI reader import"
python -c "import aicspylibczi; print('aicspylibczi OK:', aicspylibczi.__version__)"

# scripts/post_process reads transcripts.zarr.zip and always writes Parquet.
echo "Verifying transcript post-processing imports"
python -c "import zarr, pyarrow; print('zarr OK:', zarr.__version__, '| pyarrow OK:', pyarrow.__version__)"

echo "✅ Environment ${ENV_NAME} installed successfully"
echo "Activate with: conda activate ${ENV_NAME}"