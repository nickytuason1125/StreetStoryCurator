$ErrorActionPreference = 'SilentlyContinue'
$root = 'C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator'
$req  = "$root\reports\cull_request.json"
$prog = "$env:TEMP\finisher_cull.progress.jsonl"
$py   = "$root\venv\Scripts\pythonw.exe"
$runner = "$root\grade_runner.py"
Add-Type -Name K -Namespace W -MemberDefinition '[DllImport("psapi.dll")] public static extern bool EmptyWorkingSet(IntPtr h);'

function Trim-RAM {
  Get-Process | Where-Object { $_.WorkingSet64 -gt 150MB -and $_.ProcessName -ne 'pythonw' } |
    ForEach-Object { try { [W.K]::EmptyWorkingSet($_.Handle) | Out-Null } catch {} }
  Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" |
    Where-Object { $_.CommandLine -match 'StreetPhotoEditor' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
  Start-Sleep 3
}

function Last-Progress {
  $l = Get-Content $prog -Tail 1 -ErrorAction SilentlyContinue
  if (-not $l) { return $null }
  try { return ($l | ConvertFrom-Json) } catch { return $null }
}

Remove-Item $prog -ErrorAction SilentlyContinue
# single-instance lock: refuse to start if another live finisher holds it
$lock = "$root\reports\finisher.lock"
if (Test-Path $lock) {
  $lockPid = Get-Content $lock -ErrorAction SilentlyContinue
  $peer = Get-Process -Id $lockPid -ErrorAction SilentlyContinue
  if ($peer -and $peer.ProcessName -eq 'powershell') {
    "[$(Get-Date -Format HH:mm:ss)] another finisher (PID $lockPid) is running - exiting"
    exit 0
  }
}
$PID | Set-Content $lock
# seed: start from the existing checkpoint (server's last saved state)
Start-Process -FilePath $py -ArgumentList "`"$runner`"","`"$req`"","`"$prog`"" -WorkingDirectory $root -WindowStyle Hidden
"[$(Get-Date -Format HH:mm:ss)] finisher loop started" | Tee-Object -FilePath "$root\reports\finisher.log" -Append

$lastVal = -1.0
$stuck = 0
for ($cycle = 1; $cycle -le 300; $cycle++) {
  # self-heal: kill rival finisher instances, re-assert lock
  $me = $PID
  Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.ProcessId -ne $me -and $_.CommandLine -match 'cull_finisher' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  $me | Set-Content $lock
  Start-Sleep 25
  $lp = Last-Progress
  $running = (Get-Process pythonw -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*Python312*' }).Count
  $stale = $false
  if ($prog -and ((Get-Date) - (Get-Item $prog).LastWriteTime).TotalSeconds -gt 150) { $stale = $true }
  if ($lp -and $lastVal -eq $lp.progress) { $stuck++ } else { $stuck = 0; $lastVal = $lp.progress }
  $wedged = ($stuck -ge 5)

  if ($lp -and $lp.progress -ge 0.999) {
    "[$(Get-Date -Format HH:mm:ss)] CULL COMPLETE" | Tee-Object -FilePath "$root\reports\finisher.log" -Append
    break
  }
  if (($lp -and $lp.error) -or $stale -or $wedged -or $running -eq 0) {
    $why = if ($lp.error) { 'error entry' } elseif ($stale) { 'stale progress' } elseif ($wedged) { 'stuck value' } else { 'no procs' }
    "[$(Get-Date -Format HH:mm:ss)] cycle $cycle died ($why) at p=$($lp.progress) - trimming + relaunch" |
      Tee-Object -FilePath "$root\reports\finisher.log" -Append
    Trim-RAM
    Remove-Item $prog -ErrorAction SilentlyContinue
    $stuck = 0
    $lastVal = -1.0
    Start-Process -FilePath $py -ArgumentList "`"$runner`"","`"$req`"","`"$prog`"" -WorkingDirectory $root -WindowStyle Hidden
  } else {
    "[$(Get-Date -Format HH:mm:ss)] running p=$($lp.progress)" | Tee-Object -FilePath "$root\reports\finisher.log" -Append
  }
}
# restart backend + report completion
Start-Process -FilePath $py -ArgumentList 'src\local_launcher.py','--server-only' -WorkingDirectory $root -WindowStyle Hidden
"[$(Get-Date -Format HH:mm:ss)] finisher exiting - backend restarted" | Tee-Object -FilePath "$root\reports\finisher.log" -Append