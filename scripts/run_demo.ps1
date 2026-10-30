$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$demoHome = Join-Path $root ".demo-home"
$streamlitDir = Join-Path $demoHome ".streamlit"
$mplCache = Join-Path $root ".mpl-cache"

New-Item -ItemType Directory -Force $streamlitDir | Out-Null
New-Item -ItemType Directory -Force $mplCache | Out-Null

$env:USERPROFILE = $demoHome
$env:HOME = $demoHome
$env:MPLCONFIGDIR = $mplCache
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$env:PYTHONIOENCODING = "utf-8"

. (Join-Path $root "scripts\activate.ps1")
Push-Location $root
try {
    streamlit run app\streamlit_app.py @args
}
finally {
    Pop-Location
}
