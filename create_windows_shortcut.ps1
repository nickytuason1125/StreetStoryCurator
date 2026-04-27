# Creates a one-click desktop shortcut for Street Story Curator.
# Usage: right-click and run in PowerShell, or run from an elevated PowerShell prompt if required.

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$shortcutName = 'Street Story Curator.lnk'
$desktopPath = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktopPath $shortcutName

$targetPath = Join-Path $env:WINDIR 'System32\wscript.exe'
$arguments = '"' + Join-Path $scriptDir 'launch_hidden.vbs' + '"'

$wshShell = New-Object -ComObject WScript.Shell
$shortcut = $wshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetPath
$shortcut.Arguments = $arguments
$shortcut.WorkingDirectory = $scriptDir
$shortcut.IconLocation = "$targetPath,0"
$shortcut.Description = 'Launch Street Story Curator locally with embedded pywebview'
$shortcut.Save()

Write-Host "Created desktop shortcut: $shortcutPath"