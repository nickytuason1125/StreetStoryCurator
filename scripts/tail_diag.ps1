$log = 'C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator\reports\diag_run.log'
$out = 'C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator\reports\diag_tail.txt'
"--- tail @ $(Get-Date -Format HH:mm:ss) ---" | Set-Content $out
Get-Content $log -Tail 15 -ErrorAction SilentlyContinue | Add-Content $out
$pf = "$env:TEMP\diag_cull.progress.jsonl"
if (Test-Path $pf) {
  ("progress: " + (Get-Content $pf -Tail 1)) | Add-Content $out
  ("pf mtime: " + (Get-Item $pf).LastWriteTime.ToString('HH:mm:ss')) | Add-Content $out
}
$rr = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'grade_runner' })
("runners alive: " + $rr.Count) | Add-Content $out
$os = Get-CimInstance Win32_OperatingSystem
("free RAM: " + [math]::Round($os.FreePhysicalMemory/1MB,2) + " GB") | Add-Content $out