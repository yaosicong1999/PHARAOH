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
pip install --upgrade pip setuptools wheel

echo "Installing packages (this may take a while)..."

pip install \
aiohappyeyeballs==2.6.1 \
aiohttp==3.13.2 \
aiosignal==1.4.0 \
alabaster==1.0.0 \
anndata==0.11.4 \
annotated-types==0.7.0 \
app-model==0.4.0 \
appdirs==1.4.4 \
appnope==0.1.4 \
array-api-compat==1.12.0 \
asciitree==0.3.3 \
asttokens==3.0.1 \
async-timeout==5.0.1 \
attrs==25.4.0 \
babel==2.17.0 \
bermuda==0.1.6 \
Brotli==1.0.9 \
build==1.3.0 \
cachey==0.2.1 \
cairocffi==1.7.1 \
CairoSVG==2.8.2 \
certifi==2025.11.12 \
cffi==2.0.0 \
charset-normalizer==3.4.4 \
click==8.3.1 \
cloudpickle==3.1.2 \
comm==0.2.3 \
contourpy==1.3.1 \
cssselect2==0.8.0 \
cycler==0.11.0 \
dask==2025.11.0 \
debugpy==1.8.17 \
decorator==5.2.1 \
defusedxml==0.7.1 \
docstring_parser==0.17.0 \
docutils==0.21.2 \
exceptiongroup==1.3.1 \
executing==2.2.1 \
fasteners==0.20 \
flexcache==0.3 \
flexparser==0.4 \
fonttools==4.55.3 \
freetype-py==2.5.1 \
frozenlist==1.8.0 \
fsspec==2025.10.0 \
h5py==3.15.1 \
HeapDict==1.0.1 \
hsluv==5.0.4 \
idna==3.11 \
imagecodecs==2024.9.22 \
imageio==2.37.0 \
imagesize==1.4.1 \
importlib_metadata==8.7.0 \
in-n-out==0.2.1 \
ipykernel==6.31.0 \
ipython==8.37.0 \
ipython_pygments_lexers==1.1.1 \
jedi==0.19.2 \
Jinja2==3.1.6 \
joblib==1.5.1 \
jsonschema==4.25.1 \
jsonschema-specifications==2025.9.1 \
jupyter_client==8.6.3 \
jupyter_core==5.9.1 \
kiwisolver==1.4.8 \
lazy_loader==0.4 \
legacy-api-wrap==1.5 \
llvmlite==0.45.1 \
locket==1.0.0 \
magicgui==0.10.1 \
markdown-it-py==4.0.0 \
MarkupSafe==3.0.3 \
matplotlib==3.10.0 \
matplotlib-inline==0.2.1 \
mdurl==0.1.2 \
multidict==6.7.0 \
napari==0.6.6 \
napari-console==0.1.4 \
napari-plugin-engine==0.2.0 \
napari-plugin-manager==0.1.8 \
napari-svg==0.2.1 \
natsort==8.4.0 \
nest-asyncio==1.6.0 \
networkx==3.4.2 \
npe2==0.7.9 \
numba==0.62.1 \
numcodecs==0.13.1 \
numpy==2.2.6 \
numpydoc==1.9.0 \
ome-types==0.6.3 \
opencv-python==4.12.0.88 \
packaging==25.0 \
pandas==2.3.3 \
parso==0.8.5 \
partd==1.4.2 \
patsy==1.0.2 \
pexpect==4.9.0 \
pillow==11.3.0 \
Pint==0.24.4 \
platformdirs==4.5.0 \
pooch==1.8.2 \
prompt_toolkit==3.0.52 \
propcache==0.4.1 \
psutil==7.1.3 \
psygnal==0.15.0 \
ptyprocess==0.7.0 \
pure_eval==0.2.3 \
pyconify==0.2.1 \
pycparser==3.0 \
pydantic==2.12.4 \
pydantic-compat==0.1.2 \
pydantic_core==2.41.5 \
pydantic-extra-types==2.11.0 \
Pygments==2.19.2 \
pynndescent==0.5.13 \
PyOpenGL==3.1.10 \
pyparsing==3.2.0 \
pyproject_hooks==1.2.0 \
PyQt5==5.15.11 \
PyQt5-Qt5==5.15.18 \
PyQt5_sip==12.17.1 \
PyQt6==6.10.0 \
PyQt6-Qt6==6.10.0 \
PyQt6_sip==13.10.2 \
python-dateutil==2.9.0.post0 \
pytz==2025.2 \
pyvips==3.1.1 \
PyYAML==6.0.3 \
pyzmq==27.1.0 \
qtconsole==5.7.0 \
QtPy==2.4.3 \
referencing==0.37.0 \
requests==2.32.5 \
rich==14.2.0 \
rpds-py==0.29.0 \
scanpy==1.11.5 \
scikit-image==0.25.2 \
scikit-learn==1.7.2 \
scipy==1.15.3 \
seaborn==0.13.2 \
session-info2==0.2.3 \
shapely==2.1.1 \
shellingham==1.5.4 \
six==1.17.0 \
snowballstemmer==3.0.1 \
Sphinx==8.1.3 \
sphinxcontrib-applehelp==2.0.0 \
sphinxcontrib-devhelp==2.0.0 \
sphinxcontrib-htmlhelp==2.1.0 \
sphinxcontrib-jsmath==1.0.1 \
sphinxcontrib-qthelp==2.0.0 \
sphinxcontrib-serializinghtml==2.0.0 \
stack-data==0.6.3 \
statsmodels==0.14.5 \
superqt==0.7.6 \
threadpoolctl==3.5.0 \
tifffile==2025.5.10 \
tinycss2==1.5.1 \
tomli==2.3.0 \
tomli_w==1.2.0 \
toolz==1.1.0 \
tornado==6.5.1 \
tqdm==4.67.1 \
traitlets==5.14.3 \
triangle==20230923 \
typer==0.20.0 \
typing_extensions==4.15.0 \
typing-inspection==0.4.2 \
tzdata==2025.2 \
umap-learn==0.5.9.post2 \
unicodedata2==15.1.0 \
urllib3==2.5.0 \
vispy==0.15.2 \
wcwidth==0.2.14 \
webencodings==0.5.1 \
wrapt==2.0.1 \
xsdata==26.1 \
yarl==1.22.0 \
zarr==2.18.3 \
zipp==3.23.0

echo "✅ Environment ${ENV_NAME} installed successfully"
echo "Activate with: conda activate ${ENV_NAME}"