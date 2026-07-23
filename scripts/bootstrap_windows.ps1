<#
.SYNOPSIS
  Fresh Windows PC: install Git + Miniconda (if missing), clone repo, create conda env.

.DESCRIPTION
  Run from any folder on a new machine. Does NOT require the repo to exist yet.

.EXAMPLE
  # Download script only (no git yet):
  Invoke-WebRequest -Uri "https://raw.githubusercontent.com/MosheVB/3D-Scanner/main/scripts/bootstrap_windows.ps1" -OutFile "$env:USERPROFILE\bootstrap_windows.ps1"
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
  & "$env:USERPROFILE\bootstrap_windows.ps1"

.EXAMPLE
  # After clone:
  cd $env:USERPROFILE\Code\3D-Scanner
  .\scripts\bootstrap_windows.ps1
#>
param(
    [string]$CodeDir = "$env:USERPROFILE\Code",
    [string]$RepoName = "3D-Scanner",
    [string]$RepoUrl = "https://github.com/MosheVB/3D-Scanner.git",
    [string]$CondaBase = "$env:USERPROFILE\Miniconda3",
    [switch]$SkipClone,
    [switch]$SkipWinget
)

$ErrorActionPreference = "Stop"
$RepoRoot = Join-Path $CodeDir $RepoName

function Write-Step([string]$Msg) {
    Write-Host ""
    Write-Host "==> $Msg" -ForegroundColor Cyan
}

function Refresh-Path {
    $machine = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Find-Git {
    $cmd = Get-Command git -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "${env:ProgramFiles}\Git\cmd\git.exe",
        "${env:ProgramFiles(x86)}\Git\cmd\git.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Find-Conda {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $p = Join-Path $CondaBase "Scripts\conda.exe"
    if (Test-Path $p) { return $p }
    return $null
}

function Install-WithWinget([string]$Id, [string]$Label) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        Write-Host "winget not found. Install $Label manually, then re-run this script." -ForegroundColor Yellow
        return $false
    }
    Write-Host "Installing $Label via winget ($Id) ..."
    & winget install --id $Id -e --source winget --accept-package-agreements --accept-source-agreements
    Refresh-Path
    return $true
}

Write-Step "Create project folder: $CodeDir"
New-Item -ItemType Directory -Force -Path $CodeDir | Out-Null

if (-not $SkipWinget) {
    Refresh-Path

    if (-not (Find-Git)) {
        Write-Step "Git not found"
        if (-not (Install-WithWinget "Git.Git" "Git for Windows")) {
            Write-Host "Download: https://git-scm.com/download/win" -ForegroundColor Yellow
            Write-Host "After install, close PowerShell, open a new window, re-run this script."
            exit 1
        }
        Start-Sleep -Seconds 2
        Refresh-Path
    }

    if (-not (Find-Conda)) {
        Write-Step "Miniconda not found"
        if (-not (Install-WithWinget "Anaconda.Miniconda3" "Miniconda3")) {
            Write-Host "Download: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -ForegroundColor Yellow
            Write-Host "Install to: $CondaBase (default is fine). Re-run this script after install."
            exit 1
        }
        Start-Sleep -Seconds 3
        Refresh-Path
    }
}

Refresh-Path
$git = Find-Git
if (-not $git) {
    Write-Host "Git still not on PATH. Close PowerShell, open a NEW window, run:" -ForegroundColor Red
    Write-Host "  & `"$PSCommandPath`""
    exit 1
}

$conda = Find-Conda
if (-not $conda) {
    Write-Host "Conda still not on PATH. Close PowerShell, open a NEW window, run:" -ForegroundColor Red
    Write-Host "  & `"$PSCommandPath`""
    exit 1
}

$condaRoot = Split-Path (Split-Path $conda -Parent) -Parent
$env:PATH = "$condaRoot;$condaRoot\Scripts;$condaRoot\Library\bin;$env:PATH"

Write-Step "Configure Git (credential manager for GitHub login prompts)"
& $git config --global credential.helper manager 2>$null
& $git config --global init.defaultBranch main 2>$null
Write-Host "On first clone, Git will prompt for GitHub sign-in (browser or PAT)."

if (-not $SkipClone) {
    Write-Step "Clone repository"
    if (Test-Path $RepoRoot) {
        Write-Host "Already exists: $RepoRoot (skipping clone)"
    } else {
        & $git clone $RepoUrl $RepoRoot
    }
}

if (-not (Test-Path $RepoRoot)) {
    Write-Host "Repo not found at $RepoRoot" -ForegroundColor Red
    exit 1
}

Write-Step "Create conda environment (3d-scanner)"
$EnvName = "3d-scanner"
$condaExe = Join-Path $condaRoot "Scripts\conda.exe"
if (-not (Test-Path $condaExe)) {
    throw "conda.exe not found at $condaExe"
}

& $condaExe config --set channel_priority strict
$envList = & $condaExe env list 2>&1 | Out-String
if ($envList -notmatch "\b$EnvName\b") {
    & $condaExe create -n $EnvName python=3.11 pip -c conda-forge --override-channels -y
}
& $condaExe run -n $EnvName conda install open3d -c conda-forge --override-channels -y
& $condaExe run -n $EnvName pip install -r (Join-Path $RepoRoot "requirements.txt")

$setupScript = Join-Path $RepoRoot "scripts\setup_windows.ps1"
if (Test-Path $setupScript) {
    Write-Host "(setup_windows.ps1 also present — env already configured above)"
}

Write-Step "Done"
Write-Host @"

Next steps (every new PowerShell window):

  `$env:PATH = "$condaRoot;$condaRoot\Scripts;$condaRoot\Library\bin;`$env:PATH"
  conda activate 3d-scanner
  cd "$RepoRoot"
  python -m scanner --help

RealSense diagnostics:

  python scripts/realsense_probe.py
  python scripts/realsense_exposure_diag.py --stream --reset

Autonomous D405 tuning (no mouse — leave scene as-is):

  .\scripts\start_autotune.ps1
  # live view: agent_runs\autotune_live\latest.jpg

Tip: If 'python' opens the Microsoft Store, use 'conda activate 3d-scanner' first,
or disable App execution aliases for python.exe in Windows Settings.
"@
