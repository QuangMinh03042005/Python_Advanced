$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvActivate = Join-Path $root ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvActivate)) {
    throw "Không tìm thấy .venv. Hãy tạo môi trường trước: py -3.12 -m venv .venv"
}

. $venvActivate

$env:HF_HOME = Join-Path $root ".hf-cache"
$env:HF_HUB_DISABLE_SYMLINKS = "1"
$env:UV_CACHE_DIR = Join-Path $root ".uv-cache"
$env:RUFF_CACHE_DIR = Join-Path $root ".ruff-cache"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Đã kích hoạt .venv"
Write-Host "Python: $((Get-Command python).Source)"
Write-Host "HF_HOME: $env:HF_HOME"
python --version
