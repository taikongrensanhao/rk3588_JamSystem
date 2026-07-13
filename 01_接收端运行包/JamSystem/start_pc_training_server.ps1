$ErrorActionPreference = "Stop"

$BaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = $env:JAMSYSTEM_PC_PYTHON
if (-not $Python) {
    $Python = "E:\miniconda3\envs\learnBP\python.exe"
}

$HostArg = if ($env:JAMSYSTEM_SERVER_HOST) { $env:JAMSYSTEM_SERVER_HOST } else { "0.0.0.0" }
$PortArg = if ($env:JAMSYSTEM_SERVER_PORT) { $env:JAMSYSTEM_SERVER_PORT } else { "8008" }
$SimDataDir = if ($env:JAMSYSTEM_SIM_DATA_DIR) {
    $env:JAMSYSTEM_SIM_DATA_DIR
} elseif ($env:MIX_SIM_DATA_DIR) {
    $env:MIX_SIM_DATA_DIR
} else {
    "E:\桌面迁移\无线电干扰识别与复原交付\干扰样式识别流程图及参考代码"
}

$DataDir = if ($env:JAMSYSTEM_REALTIME_DATA_DIR) { $env:JAMSYSTEM_REALTIME_DATA_DIR } else { Join-Path $BaseDir "dataset_realtime" }
$BaseDataDir = if ($env:JAMSYSTEM_REALWORLD_DATA_DIR) { $env:JAMSYSTEM_REALWORLD_DATA_DIR } else { Join-Path $BaseDir "dataset_realworld" }
$Epochs = if ($env:JAMSYSTEM_TRAIN_EPOCHS) { $env:JAMSYSTEM_TRAIN_EPOCHS } else { "3" }
$BatchSize = if ($env:JAMSYSTEM_TRAIN_BATCH) { $env:JAMSYSTEM_TRAIN_BATCH } else { "16" }
$RknnCommand = $env:JAMSYSTEM_RKNN_COMMAND
$EnableRknnConvert = if ($env:JAMSYSTEM_ENABLE_RKNN_CONVERT) { $env:JAMSYSTEM_ENABLE_RKNN_CONVERT } else { "1" }
if (-not $RknnCommand -and $EnableRknnConvert -ne "0") {
    $WslRknnScript = Join-Path $BaseDir "convert_latest_model_to_rknn_wsl.ps1"
    if (Test-Path $WslRknnScript) {
        $RknnCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\convert_latest_model_to_rknn_wsl.ps1"
    } else {
        $AutoRknnScript = Join-Path $BaseDir "convert_latest_model_to_rknn.py"
        $RknnCommand = "`"$Python`" `"$AutoRknnScript`""
    }
}

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
New-Item -ItemType Directory -Force -Path $BaseDataDir | Out-Null

Write-Host "[SERVER] python: $Python"
Write-Host "[SERVER] listen: ${HostArg}:${PortArg}"
Write-Host "[SERVER] base data dir: $BaseDataDir"
Write-Host "[SERVER] realtime data dir: $DataDir"
Write-Host "[SERVER] sim data dir: $SimDataDir"
Write-Host "[SERVER] epochs: $Epochs batch: $BatchSize"
if ($RknnCommand) {
    Write-Host "[SERVER] rknn command: $RknnCommand"
}

Set-Location $BaseDir
$ServerArgs = @(
    "server_realtime_train.py",
    "--host", $HostArg,
    "--port", $PortArg,
    "--base-data-dir", $BaseDataDir,
    "--data-dir", $DataDir,
    "--sim-data-dir", $SimDataDir,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize
)
if ($RknnCommand) {
    $ServerArgs += @("--rknn-command", $RknnCommand)
}
& $Python @ServerArgs
