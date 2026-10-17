# ============================================================
# install_conda_env.ps1 - Windows counterpart of install_conda_env.sh
#
#   powershell -ExecutionPolicy Bypass -File install_conda_env.ps1
#   powershell -ExecutionPolicy Bypass -File install_conda_env.ps1 -InstallMiniforge
#
# -InstallMiniforge is for a machine with no conda at all: Miniforge is downloaded from
# its GitHub release and installed silently for this user to %USERPROFILE%\miniforge3
# (no PATH change, not registered as the system Python), then the environment is built
# in it. Without the switch, a machine with no conda stops with a message.
#
# Same environment name and the same pins as install_conda_env.sh. Two steps differ,
# both because of Windows, not PHARAOH:
#   * libvips comes from conda-forge together with Python. pyvips is only a binding,
#     and on Windows the library it binds is a DLL that must be on the search path.
#     If `import pyvips` still fails, the wheel that bundles libvips
#     (pyvips[binary]) is tried.
#   * imagecodecs and aicspylibczi are installed one at a time, after the main list.
#     They are the pins most likely to have no Windows wheel for this Python, and a
#     failure inside the bulk install would leave nothing installed. imagecodecs is
#     required (it decodes the JPEG 2000 Xenium DAPI). aicspylibczi only reads .czi
#     input, so OME-TIFF input works without it.
# conda does not have to be on PATH: the script also looks in CONDA_EXE and the usual
# install folders (miniforge3 / miniconda3 / anaconda3 under the user profile,
# LocalAppData, ProgramData, C:\). Use version_5.0_windows\*.ps1 to run PHARAOH on
# Windows.
# The environment is solved from conda-forge only (--override-channels): on a Miniconda
# or Anaconda install the default channels would otherwise be consulted too, and recent
# conda stops non-interactively until their terms of service are accepted.
# ============================================================
param(
    [switch]$InstallMiniforge
)
$ErrorActionPreference = "Stop"

$EnvName = "PHARAOH"
$PythonVersion = "3.10"

# conda.exe from PATH, CONDA_EXE, or a standard install folder. A plain PowerShell
# window only has `conda` after `conda init powershell`, and many installs never ran it.
function Find-CondaExe {
    $found = Get-Command conda.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { return $found.Source }
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) { return $env:CONDA_EXE }
    foreach ($base in @($env:USERPROFILE, $env:LOCALAPPDATA, $env:ProgramData, "C:\")) {
        if (-not $base) { continue }
        foreach ($name in @("miniforge3", "Miniforge3", "mambaforge", "miniconda3", "Miniconda3", "anaconda3", "Anaconda3")) {
            $candidate = Join-Path (Join-Path $base $name) "Scripts\conda.exe"
            if (Test-Path $candidate) { return $candidate }
        }
    }
    return $null
}

# Silent per-user Miniforge install; returns its conda.exe. Same steps as
# revision\windows\install_baselines_windows.ps1.
function Install-Miniforge {
    param([string]$Target)
    $url = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
    $installer = Join-Path $env:TEMP "Miniforge3-Windows-x86_64.exe"
    $ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is ~10x slower with it on 5.1
    Write-Host "No conda found. Downloading Miniforge: $url"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
    $mb = [math]::Round((Get-Item $installer).Length / 1MB, 1)
    if ($mb -lt 20) { throw "downloaded installer is only $mb MB, which is not a Miniforge installer" }
    Write-Host "Installing Miniforge ($mb MB) to $Target silently (about a minute)"
    # /D= must be the LAST argument and unquoted: the NSIS installer takes the rest of
    # the command line as the path, spaces included.
    $proc = Start-Process -FilePath $installer -Wait -PassThru `
        -ArgumentList "/S /InstallationType=JustMe /RegisterPython=0 /D=$Target"
    if ($proc.ExitCode -ne 0) { throw "Miniforge installer exited with $($proc.ExitCode)" }
    Remove-Item $installer -ErrorAction SilentlyContinue
    $installed = Join-Path $Target "Scripts\conda.exe"
    if (-not (Test-Path $installed)) { throw "Miniforge reported success but $installed is not there" }
    return $installed
}

$conda = Find-CondaExe
if (-not $conda -and $InstallMiniforge) {
    $conda = Install-Miniforge (Join-Path $env:USERPROFILE "miniforge3")
}
if (-not $conda) {
    Write-Host "conda was not found (not on PATH, no CONDA_EXE, not in the usual install folders)." -ForegroundColor Red
    Write-Host "Run this script again with -InstallMiniforge to install Miniforge for this user," -ForegroundColor Red
    Write-Host "or install it yourself (https://conda-forge.org/download/), or set `$env:CONDA_EXE" -ForegroundColor Red
    Write-Host "to the full path of conda.exe first." -ForegroundColor Red
    exit 127
}
Write-Host "Using conda: $conda"

# Runs a native command and stops the script if it fails; PowerShell does not stop on
# a non-zero exit code by itself.
function Invoke-Step {
    param([string]$File, [string[]]$Arguments)
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$File $($Arguments -join ' ') exited with $LASTEXITCODE" }
}

# Runs python inside the environment, like `conda activate` + python in the .sh.
function Invoke-EnvPython {
    param([string[]]$Arguments, [switch]$AllowFailure)
    & $conda run --no-capture-output -n $EnvName python @Arguments
    if ($LASTEXITCODE -ne 0 -and -not $AllowFailure) {
        throw "python $($Arguments -join ' ') exited with $LASTEXITCODE"
    }
    return
}

Write-Host "Creating conda environment: $EnvName (with libvips from conda-forge)"
Invoke-Step $conda @("create", "-y", "-n", $EnvName, "--override-channels", "-c", "conda-forge", "python=$PythonVersion", "libvips")

Write-Host "Upgrading pip"
Invoke-EnvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

Write-Host "Installing packages (this may take a while)..."
Invoke-EnvPython @("-m", "pip", "install",
    "numpy==2.2.6",
    "pandas==2.3.3",
    "scipy==1.15.3",
    "matplotlib==3.10.0",
    "pillow==11.3.0",
    "opencv-python==4.12.0.88",
    "tifffile==2025.5.10",
    "ome-types==0.6.3",
    "pyvips==3.1.1",
    "shapely==2.1.1",
    "scikit-image==0.25.2",
    "scikit-learn==1.7.2",
    "tqdm==4.67.1",
    "psutil==7.2.2",
    "PyQt5==5.15.11",
    "zarr",
    "pyarrow")

Write-Host "Installing imagecodecs (required: JPEG 2000 DAPI)"
Invoke-EnvPython @("-m", "pip", "install", "imagecodecs==2024.9.22")

Write-Host "Installing aicspylibczi (optional: .czi input only)"
Invoke-EnvPython @("-m", "pip", "install", "aicspylibczi==4.0.0") -AllowFailure
if ($LASTEXITCODE -ne 0) {
    Write-Host "!! aicspylibczi==4.0.0 has no wheel here; .czi input is unavailable, OME-TIFF is not affected" -ForegroundColor Yellow
}

Write-Host "Verifying OpenCV import"
Invoke-EnvPython @("-c", "import cv2; print('cv2 OK:', cv2.__version__)")

Write-Host "Verifying pyvips / libvips"
Invoke-EnvPython @("-c", "import pyvips; print('pyvips OK:', pyvips.__version__)") -AllowFailure
if ($LASTEXITCODE -ne 0) {
    Write-Host "pyvips could not load libvips; installing pyvips[binary], which bundles it" -ForegroundColor Yellow
    Invoke-EnvPython @("-m", "pip", "install", "--force-reinstall", "pyvips[binary]")
    Invoke-EnvPython @("-c", "import pyvips; print('pyvips OK:', pyvips.__version__)")
}

Write-Host "Verifying imagecodecs and psutil imports"
Invoke-EnvPython @("-c", "import imagecodecs, psutil; print('imagecodecs OK:', imagecodecs.__version__, '| psutil OK:', psutil.__version__)")

# scripts/post_process reads transcripts.zarr.zip and always writes Parquet.
Write-Host "Verifying transcript post-processing imports"
Invoke-EnvPython @("-c", "import zarr, pyarrow; print('zarr OK:', zarr.__version__, '| pyarrow OK:', pyarrow.__version__)")

Write-Host "Environment $EnvName installed successfully" -ForegroundColor Green
Write-Host "Activate with: conda activate $EnvName"
