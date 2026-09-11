$out = 'C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator\reports\state.txt'
"now: $(Get-Date -Format HH:mm:ss)" | Set-Content $out
$r = @(Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" | Where-Object { $_.CommandLine -match 'grade_runner' })
"runners: $($r.Count)" | Add-Content $out
$r | ForEach-Object { "runner $($_.ProcessId) started $($_.CreationDate.ToString('HH:mm:ss'))" } | Add-Content $out
Get-Process powershell -ErrorAction SilentlyContinue | ForEach-Object { "ps $($_.Id) started $($_.StartTime.ToString('HH:mm:ss'))" } | Add-Content $out
"lock: $(Test-Path 'C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator\reports\finisher.lock')" | Add-Content $out
$pf = "$env:TEMP\finisher_cull.progress.jsonl"
if (Test-Path $pf) { (Get-Content $pf -Tail 1) + " @ $((Get-Item $pf).LastWriteTime.ToString('HH:mm:ss'))" | Add-Content $out }