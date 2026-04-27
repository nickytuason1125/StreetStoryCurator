$root = "C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator"
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut("$env:USERPROFILE\Desktop\Street Story Curator.lnk")
$sc.TargetPath = "wscript.exe"
$sc.Arguments = "`"$root\launch_hidden.vbs`""
$sc.WorkingDirectory = $root
$sc.IconLocation = "$root\frontend\src-tauri\icons\icon.ico,0"
$sc.Description = "Street Story Curator (local pywebview launcher)"
$sc.Save()
Write-Host "Shortcut created on Desktop"