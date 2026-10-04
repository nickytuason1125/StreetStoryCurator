# dev.ps1 - the no-dual-tracking development loop.
#
#   scripts\dev.ps1
#
# Starts the backend with --reload (Python edits hot-reload) and prints the
# one command to run in a second terminal for live UI edits. In this mode the
# app is served by Vite on :5173 from SOURCE, so `npm run build` is only
# needed before packaging/shipping - not to see your changes.

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "== FirstCut dev mode ==" -ForegroundColor Cyan
Write-Host "Backend : http://127.0.0.1:8000  (--reload: .py edits hot-reload)"
Write-Host "UI      : run 'npm run dev' in frontend\ -> http://localhost:5173"
Write-Host "          (Vite proxies /api to the backend; source edits are live)"
Write-Host "Ship    : 'npm run build' in frontend\, then restart the backend"
Write-Host ""

Push-Location $root
try {
    & $py -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
} finally {
    Pop-Location
}
