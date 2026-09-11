Add-Type -Name K -Namespace W -MemberDefinition '[DllImport("psapi.dll")] public static extern bool EmptyWorkingSet(IntPtr h);'
$before = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)
Get-Process | Where-Object { $_.WorkingSet64 -gt 150MB -and $_.ProcessName -ne 'pythonw' } | ForEach-Object {
  try { [W.K]::EmptyWorkingSet($_.Handle) | Out-Null } catch {}
}
Start-Sleep 3
$after = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)
"free RAM before: $before GB -> after: $after GB"
Get-Process pythonw -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep 2
$after2 = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)
"after killing stalled grade procs: $after2 GB free"