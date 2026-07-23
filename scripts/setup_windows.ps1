<#
.SYNOPSIS
  One-time setup for Windows: Miniconda + 3d-scanner conda env.

.EXAMPLE
  cd C:\Users\You\Code\3D-Scanner
  .\scripts\setup_windows.ps1
#>
param(
    [string]$CondaBase = "$env:USERPROFILE\Miniconda3"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$EnvName = "3d-scanner"

function Ensure-CondaOnPath {
    param([string]$Base)
    $condaExe = Join-Path $Base "Scripts\conda.exe"
    if (-not (Test-Path $condaExe)) {
        throw "conda not found at $condaExe. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html"
    }
    $env:PATH = "$Base;$Base\Scripts;$Base\Library\bin;$env:PATH"
}

if (-not (Test-Path (Join-Path $CondaBase "Scripts\conda.exe"))) {
    Write-Host "Miniconda not found at $CondaBase"
    Write-Host "Download and install: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    Write-Host "Then re-run: .\scripts\setup_windows.ps1"
    exit 1
}

Ensure-CondaOnPath -Base $CondaBase
conda config --set channel_priority strict

$envList = conda env list 2>&1 | Out-String
if ($envList -notmatch "\b$EnvName\b") {
    conda create -n $EnvName python=3.11 pip -c conda-forge --override-channels -y
}

conda run -n $EnvName conda install open3d -c conda-forge --override-channels -y
conda run -n $EnvName pip install -r (Join-Path $RepoRoot "requirements.txt")

Write-Host ""
Write-Host "Done. Activate with:"
Write-Host "  `$env:PATH = `"$CondaBase;$CondaBase\Scripts;$CondaBase\Library\bin;`$env:PATH`""
Write-Host "  conda activate $EnvName"
Write-Host "  cd `"$RepoRoot`""
Write-Host "  python -m scanner --help"
