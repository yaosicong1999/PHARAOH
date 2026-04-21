#!/usr/bin/env bash
set -e

ENV_NAME="PHARAOH"
PYTHON_VERSION="3.10"

echo "Creating conda environment: ${ENV_NAME}"
conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

pip install --upgrade pip setuptools wheel

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
triangle==20230923 \
PyQt5==5.15.11 \

echo "✅ Environment ${ENV_NAME} installed successfully"
echo "Activate with: conda activate ${ENV_NAME}"