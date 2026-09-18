Dim shell, fso, appDir
Set shell = CreateObject("WScript.Shell")
Set fso   = CreateObject("Scripting.FileSystemObject")
appDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))

shell.CurrentDirectory = Left(appDir, Len(appDir) - 1)

' Tier policy (2026-09-13): NO pin here. The tier is now LIBRARY STATE —
' src/library_tier.py persists the first choice (or the user's explicit
' pick) and every later grade inherits it, so adding photos no longer
' re-encodes the library and the quality never silently drops between
' runs. Lite remains an EXPLICIT choice: Start-Lite.bat (or Start-Lite.sh)
' still sets SIGLIP_TIER=low / FIRSTCUT_LITE=1 for its session. Removing
' the old hard pin here is what stops "the same grade came out worse than
' last time" — see LITE_MODE.md and src/library_tier.py.

Dim pythonwPath, scriptPath, crashLog, stampFile
pythonwPath = appDir & "venv\Scripts\pythonw.exe"
scriptPath  = appDir & "src\local_launcher.py"
crashLog    = appDir & "crash.log"
stampFile   = appDir & "venv\.setup_ok"

' Kill any leftover pythonw process. NOTE: msedgewebview2.exe is deliberately
' NOT killed — force-killing WebView2 made Edge pop recovery tabs on next launch.
shell.Run "cmd /c taskkill /F /FI ""IMAGENAME eq pythonw.exe"" >nul 2>&1", 0, True
On Error Resume Next
If fso.FileExists(crashLog) Then fso.DeleteFile crashLog, True
On Error GoTo 0

' If venv missing or stamp absent, run Start.bat to (re)install
If Not fso.FileExists(pythonwPath) Or Not fso.FileExists(stampFile) Then
    shell.Run "cmd /c call Start.bat", 1, True
    WScript.Quit
End If

' Launch immediately — relative paths, no spaces, no quoting issues
shell.Run "venv\Scripts\pythonw.exe src\local_launcher.py", 0, False
