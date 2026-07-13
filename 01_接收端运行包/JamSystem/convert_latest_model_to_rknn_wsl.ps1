$ErrorActionPreference = "Stop"

$BaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Distro = if ($env:JAMSYSTEM_WSL_DISTRO) { $env:JAMSYSTEM_WSL_DISTRO } else { "Ubuntu" }
$Venv = if ($env:JAMSYSTEM_WSL_RKNN_VENV) { $env:JAMSYSTEM_WSL_RKNN_VENV } else { '$HOME/.venvs/rknn232' }
$NoQuant = if ($env:JAMSYSTEM_RKNN_NO_QUANT) { $env:JAMSYSTEM_RKNN_NO_QUANT } else { "1" }
$SimDataDir = if ($env:JAMSYSTEM_SIM_DATA_DIR) {
    $env:JAMSYSTEM_SIM_DATA_DIR
} elseif ($env:MIX_SIM_DATA_DIR) {
    $env:MIX_SIM_DATA_DIR
} else {
    $PackageDir = Split-Path -Parent $BaseDir
    $ProjectDir = Split-Path -Parent $PackageDir
    $candidate = Get-ChildItem -Path $ProjectDir -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName "metadata.csv") } |
        Select-Object -First 1
    if (-not $candidate) {
        throw "Cannot find simulation data directory with metadata.csv under $ProjectDir"
    }
    $candidate.FullName
}

function Convert-ToWslPath([string]$Path) {
    $full = (Resolve-Path -LiteralPath $Path).Path
    if ($full -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Only Windows drive paths are supported: $Path"
    }
    $drive = $matches[1].ToLowerInvariant()
    $rest = $matches[2].Replace('\', '/')
    return "/mnt/$drive/$rest"
}

function Quote-Bash([string]$Text) {
    return "'" + $Text.Replace("'", "'\''") + "'"
}

$WslBaseDir = Convert-ToWslPath $BaseDir
$WslSimDataDir = Convert-ToWslPath $SimDataDir
$NoQuantArg = if ($NoQuant -eq "0") { "" } else { " --no-quant" }

$bash = @(
    "export LANG=C.UTF-8 LC_ALL=C.UTF-8",
    "source $Venv/bin/activate",
    "cd $(Quote-Bash $WslBaseDir)",
    "JAMSYSTEM_SIM_DATA_DIR=$(Quote-Bash $WslSimDataDir) python convert_latest_model_to_rknn.py$NoQuantArg"
) -join "; "

Write-Host "[WSL_RKNN] distro: $Distro"
Write-Host "[WSL_RKNN] base: $WslBaseDir"
Write-Host "[WSL_RKNN] sim: $WslSimDataDir"
Write-Host "[WSL_RKNN] no quant: $NoQuant"
& wsl.exe -d $Distro -- bash -lc $bash
if ($LASTEXITCODE -ne 0) {
    throw "WSL RKNN conversion failed: $LASTEXITCODE"
}
