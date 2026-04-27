# dev.ps1 — starts the backend and frontend watcher in separate terminals.
# Usage: right-click → "Run with PowerShell"  OR  pwsh -File dev.ps1

$root    = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontend = Join-Path $root "frontend"

# Backend — FastAPI
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "python server.py"

# Frontend — Vite rebuild-on-save (writes to dist/ which FastAPI serves)
Start-Process powershell -WorkingDirectory $frontend -ArgumentList "-NoExit", "-Command", "npm run watch"

Write-Host ""
Write-Host "Both terminals launched."
Write-Host "  Backend  -> http://127.0.0.1:8000"
Write-Host "  Frontend -> rebuilds dist/ automatically on every save"
Write-Host ""
Write-Host "Hard-refresh the app (Ctrl+Shift+R) after saving a .tsx file."
