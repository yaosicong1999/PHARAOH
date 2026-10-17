# ============================================================
# _pharaoh_python.ps1 - dot-sourced by the run_pharaoh_*.ps1 launchers; not run directly.
#
# Resolve-PharaohPython returns the python.exe of the PHARAOH conda environment without
# needing `conda activate` (which a PowerShell window only has after
# `conda init powershell`) or `conda run`. In order:
#   1. $env:PYTHON, if set (an explicit interpreter always wins);
#   2. `python` when that environment is already active ($env:CONDA_DEFAULT_ENV);
#   3. <conda root>\envs\<name>\python.exe, where the conda root comes from CONDA_EXE,
#      conda.exe on PATH, or the usual install folders (miniforge3 / miniconda3 /
#      anaconda3 under the user profile, LocalAppData, ProgramData, C:\).
# The environment name is $env:PHARAOH_CONDA_ENV, default PHARAOH -- what
# install_conda_env.ps1 creates. Windows paths are case-insensitive, so an environment
# named `pharaoh` is found too.
# ============================================================

function Resolve-PharaohPython {
    if ($env:PYTHON) { return $env:PYTHON }
    $envName = if ($env:PHARAOH_CONDA_ENV) { $env:PHARAOH_CONDA_ENV } else { "PHARAOH" }
    if ($env:CONDA_DEFAULT_ENV -eq $envName) { return "python" }

    $roots = @()
    if ($env:CONDA_EXE) { $roots += (Split-Path (Split-Path $env:CONDA_EXE -Parent) -Parent) }
    $onPath = Get-Command conda.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($onPath) { $roots += (Split-Path (Split-Path $onPath.Source -Parent) -Parent) }
    foreach ($base in @($env:USERPROFILE, $env:LOCALAPPDATA, $env:ProgramData, "C:\")) {
        if (-not $base) { continue }
        foreach ($name in @("miniforge3", "Miniforge3", "mambaforge", "miniconda3", "Miniconda3", "anaconda3", "Anaconda3")) {
            $roots += (Join-Path $base $name)
        }
    }
    foreach ($root in $roots) {
        $candidate = Join-Path $root "envs\$envName\python.exe"
        if (Test-Path $candidate) { return $candidate }
    }

    Write-Host "Could not find the '$envName' conda environment's python.exe." -ForegroundColor Red
    Write-Host "Build it with install_conda_env.ps1, or point to it directly, e.g." -ForegroundColor Red
    Write-Host "  `$env:PYTHON = `"C:\Users\<you>\miniforge3\envs\PHARAOH\python.exe`"" -ForegroundColor Red
    exit 127
}
