[CmdletBinding()]
param(
    [string]$StagingRoot = 'E:\comfyui\_extra\fasth3_staging',
    [string]$PortableRoot = 'D:\ComfyUI_windows_portable',
    [int]$Port = 8189,
    [switch]$Update
)

$ErrorActionPreference = 'Stop'
$stagingPath = [System.IO.Path]::GetFullPath($StagingRoot)
$portablePath = [System.IO.Path]::GetFullPath($PortableRoot)
$stagingComfy = Join-Path $stagingPath 'ComfyUI'
$portablePython = Join-Path $portablePath 'python_embeded\python.exe'
$sourceComfy = Join-Path $portablePath 'ComfyUI'

if (-not (Test-Path -LiteralPath $portablePython)) {
    throw "Portable Python not found: $portablePython"
}
if (-not (Test-Path -LiteralPath $sourceComfy)) {
    throw "Existing ComfyUI checkout not found: $sourceComfy"
}

New-Item -ItemType Directory -Force -Path $stagingPath | Out-Null
if (Test-Path -LiteralPath (Join-Path $stagingComfy '.git')) {
    if ($Update) {
        git -C $stagingComfy pull --ff-only
    }
} else {
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git $stagingComfy
}

$stagingNodes = Join-Path $stagingComfy 'custom_nodes'
New-Item -ItemType Directory -Force -Path $stagingNodes | Out-Null
foreach ($nodeName in @('ComfyUI-GGUF', 'ComfyUI-Spectrum-MiniMax-H3')) {
    $sourceNode = Join-Path $sourceComfy "custom_nodes\$nodeName"
    $targetNode = Join-Path $stagingNodes $nodeName
    if (Test-Path -LiteralPath $sourceNode) {
        Copy-Item -LiteralPath $sourceNode -Destination $targetNode -Recurse -Force
    } else {
        Write-Warning "Optional custom node not found in existing runtime: $sourceNode"
    }
}

$stagingPackages = Join-Path $stagingPath 'python_packages'
New-Item -ItemType Directory -Force -Path $stagingPackages | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $stagingPackages 'comfy_aimdo\storage.py'))) {
    & $portablePython -m pip install --disable-pip-version-check --no-input --no-deps --target $stagingPackages 'comfy-aimdo==0.5.5' 'comfy-kitchen==0.2.35'
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install isolated ComfyUI runtime dependencies"
    }
}

$extraPaths = Join-Path $stagingPath 'extra_model_paths.yaml'
$yaml = @"
legacy:
  base_path: $($portablePath -replace '\\', '/')/ComfyUI
  is_default: false
  clip: models/clip
  diffusion_models: models/diffusion_models
  unet: models/unet
  text_encoders: models/text_encoders
  vae: models/vae

fasth3:
  base_path: E:/comfyui/_extra
  is_default: true
  clip: models/clip
  diffusion_models: models/diffusion_models
  unet: models/unet
  text_encoders: models/text_encoders
  vae: models/vae
"@
Set-Content -LiteralPath $extraPaths -Value $yaml -Encoding utf8

$logOut = Join-Path $stagingPath 'comfyui.stdout.log'
$logErr = Join-Path $stagingPath 'comfyui.stderr.log'
$launcher = Join-Path $stagingPath 'launch_comfyui.py'
$launcherSource = @'
import os
import runpy
import sys

staging_root = os.path.dirname(os.path.abspath(__file__))
comfy_root = os.path.join(staging_root, 'ComfyUI')
package_root = os.path.join(staging_root, 'python_packages')
sys.path.insert(0, package_root)
sys.path.insert(0, comfy_root)
main_path = os.path.join(comfy_root, 'main.py')
sys.argv[0] = main_path
runpy.run_path(main_path, run_name='__main__')
'@
Set-Content -LiteralPath $launcher -Value $launcherSource -Encoding utf8
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output "ComfyUI staging already listens on port $Port"
    Write-Output "Root: $stagingComfy"
    Write-Output "Config: $extraPaths"
    exit 0
}

$arguments = @(
    $launcher,
    '--windows-standalone-build',
    '--listen', '127.0.0.1',
    '--port', "$Port",
    '--extra-model-paths-config', $extraPaths,
    '--disable-auto-launch'
)
$process = Start-Process -FilePath $portablePython -ArgumentList $arguments -WorkingDirectory $stagingComfy -WindowStyle Hidden -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru
Set-Content -LiteralPath (Join-Path $stagingPath 'comfyui.pid') -Value "$($process.Id)" -Encoding ascii

Write-Output "Started isolated ComfyUI staging process PID=$($process.Id)"
Write-Output "Root: $stagingComfy"
Write-Output "Config: $extraPaths"
Write-Output "Port: $Port"
