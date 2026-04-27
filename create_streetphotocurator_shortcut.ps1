# Creates a one-click desktop shortcut named StreetPhotoCurator.
# Run this from the project folder in PowerShell.

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktopPath = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktopPath 'StreetPhotoCurator.lnk'

$targetPath = Join-Path $env:WINDIR 'System32\wscript.exe'
$arguments = '"' + (Join-Path $root 'launch_hidden.vbs') + '"'
$iconPath = Join-Path $root 'frontend\src-tauri\icons\icon.ico'

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetPath
$shortcut.Arguments = $arguments
$shortcut.WorkingDirectory = $root
$shortcut.IconLocation = if (Test-Path $iconPath) { "$iconPath,0" } else { "$targetPath,0" }
$shortcut.Description = 'StreetPhotoCurator local launcher'
$shortcut.Save()

Write-Host "Created desktop shortcut: $shortcutPath"