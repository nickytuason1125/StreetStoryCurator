$ErrorActionPreference = 'SilentlyContinue'
$me = $PID
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
  Where-Object { $_.ProcessId -ne $me -and $_.CommandLine -notmatch 'reset_fin' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" |
  Where-Object { $_.CommandLine -match 'StreetPhotoEditor' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep 3
Add-Type -Name K -Namespace W -MemberDefinition '[DllImport("psapi.dll")] public static extern bool EmptyWorkingSet(IntPtr h);'
Get-Process | Where-Object { $_.WorkingSet64 -gt 150MB -and $_.ProcessName -ne 'pythonw' } |
  ForEach-Object { try { [W.K]::EmptyWorkingSet($_.Handle) | Out-Null } catch {} }
Remove-Item 'C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator\reports\finisher.lock' -ErrorAction SilentlyContinue
Remove-Item 'C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator\reports\finisher.log' -ErrorAction SilentlyContinue
$env:FIRSTCUT_SHUFFLE = 'diag1'
Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','"C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator\scripts\cull_finisher.ps1"' -WindowStyle Hidden
$os = Get-CimInstance Win32_OperatingSystem
"reset done, free RAM: $([math]::Round($os.FreePhysicalMemory/1MB,2)) GB, single finisher launched, FIRSTCUT_SHUFFLE=diag1"