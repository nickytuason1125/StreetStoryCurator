$ErrorActionPreference = 'Continue'
$root = 'C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator'
$log = "$root\deploy_engine.log"
"waiting for PyInstaller build..." | Out-File $log -Encoding utf8
$deadline = (Get-Date).AddMinutes(25)
while ((Get-Date) -lt $deadline) {
  if ((Test-Path "$root\dist\FirstCut\FirstCut.exe") -and -not (Get-Process pyinstaller -ErrorAction SilentlyContinue)) { break }
  Start-Sleep -Seconds 15
}
if (-not (Test-Path "$root\dist\FirstCut\FirstCut.exe")) { "BUILD STILL NOT DONE" | Out-File $log -Append -Encoding utf8; exit 1 }
"build done" | Out-File $log -Append -Encoding utf8

# 0. KILL FIRST â€” the running engine locks its own exe/PYZ, so copying over it
#    silently fails (locked files keep old bytes) and the "restart" just
#    relaunches the stale binary. Order is: kill â†’ copy â†’ relaunch.
Get-Process firstcut,curator-api,pythonw -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 4
"killed running app + engine" | Out-File $log -Append -Encoding utf8

# 1. fresh frontend (with banner fix + preview-first loupe) into the new engine
$fd = "$root\dist\FirstCut\_internal\frontend\dist"
Remove-Item "$fd\*" -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item "$root\frontend\dist\*" $fd -Recurse -Force
"frontend copied into engine" | Out-File $log -Append -Encoding utf8

# 2. staging copy on E:
$stage = 'E:\firstcut-engine'
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue }
Copy-Item "$root\dist\FirstCut" $stage -Recurse -Force
"staged to E:\firstcut-engine" | Out-File $log -Append -Encoding utf8

# 3. live copy where the shell expects it
$live = "$env:APPDATA\engine\curator-api-x86_64-pc-windows-msvc"
New-Item -ItemType Directory -Force -Path $live | Out-Null
Copy-Item "$root\dist\FirstCut\*" $live -Recurse -Force
Rename-Item "$live\FirstCut.exe" 'curator-api.exe' -Force
"deployed to $live" | Out-File $log -Append -Encoding utf8

# 4. relaunch + verify (app killed in step 0)
Start-Process "$env:LOCALAPPDATA\FirstCut\firstcut.exe"
Start-Sleep -Seconds 30
try { $h = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/health' -TimeoutSec 10; "health: $($h.StatusCode)" | Out-File $log -Append -Encoding utf8 } catch { "health: FAILED" | Out-File $log -Append -Encoding utf8 }
$wins = Get-Process curator-api -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -ne '' }
if ($wins) { "VISIBLE CONSOLES: $($wins.Count)" | Out-File $log -Append -Encoding utf8 } else { "no visible engine consoles" | Out-File $log -Append -Encoding utf8 }
"DONE" | Out-File $log -Append -Encoding utf8
