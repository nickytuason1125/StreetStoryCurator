$root    = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath('Desktop')
$lnk     = Join-Path $desktop 'Street Story Curator.lnk'
$icon    = Join-Path $root 'icon.ico'
$vbs     = Join-Path $root 'launch_hidden.vbs'

$sh = New-Object -ComObject WScript.Shell
$s  = $sh.CreateShortcut($lnk)
$s.TargetPath      = 'wscript.exe'
$s.Arguments       = '"' + $vbs + '"'
$s.WorkingDirectory = $root
$s.IconLocation    = $icon + ',0'
$s.Description     = 'Street Story Curator'
$s.Save()

Write-Host "Done! 'Street Story Curator' shortcut is now on your Desktop."
