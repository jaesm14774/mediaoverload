param(
    [string]$ComfyRoot = 'D:\ComfyUI_windows_portable',
    [int]$Port = 8188,
    [double]$ReserveVramGB = 1.0,
    [switch]$DynamicMode,
    [switch]$FastDisk,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $ComfyRoot 'python_embeded\python.exe'
$main = Join-Path $ComfyRoot 'ComfyUI\main.py'
$logDir = Join-Path $ComfyRoot 'logs'
$stdout = Join-Path $logDir 'comfyui-h3-lowvram.out.log'
$stderr = Join-Path $logDir 'comfyui-h3-lowvram.err.log'

if (-not (Test-Path -LiteralPath $python)) { throw "Portable Python not found: $python" }
if (-not (Test-Path -LiteralPath $main)) { throw "ComfyUI entrypoint not found: $main" }
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$reserveVramText = $ReserveVramGB.ToString([Globalization.CultureInfo]::InvariantCulture)
$arguments = @('-s', 'ComfyUI\main.py', '--windows-standalone-build', '--disable-auto-launch', '--listen', '127.0.0.1', '--port', "$Port", '--reserve-vram', $reserveVramText)
if (-not $DynamicMode) {
    $arguments += @('--disable-pinned-memory', '--disable-dynamic-vram', '--disable-async-offload', '--lowvram')
}
if ($FastDisk) {
    $arguments += '--fast-disk'
}

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output "ComfyUI already listening on port $Port (PID $($existing.OwningProcess))."
    exit 0
}

$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $ComfyRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$mode = if ($DynamicMode) { 'dynamic-VRAM + pinned-memory + async-offload' } else { 'legacy low-VRAM' }
Write-Output "Started ComfyUI H3 $mode profile with PID $($process.Id)."
Write-Output "Logs: $stdout and $stderr"

if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$Port"
}
