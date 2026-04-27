Dim shell, fso, appDir
Set shell = CreateObject("WScript.Shell")
Set fso   = CreateObject("Scripting.FileSystemObject")
appDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))

shell.CurrentDirectory = Left(appDir, Len(appDir) - 1)

Dim pythonwPath, scriptPath, crashLog, stampFile
pythonwPath = appDir & "venv\Scripts\pythonw.exe"
scriptPath  = appDir & "src\local_launcher.py"
crashLog    = appDir & "crash.log"
stampFile   = appDir & "venv\.setup_ok"

' Kill any leftover pythonw and wait for it to exit before touching crash.log
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
