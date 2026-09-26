' Silent launcher for the FMIG Rat Reconstruction app.
' Runs the GUI with pythonw.exe (no console window) using the self-contained
' "general app" environment set up by ..\setup_environment.bat. That script
' writes where it put the environment to a fixed pointer file under
' %LOCALAPPDATA%\FMIG, since it may land at C:\FMIG_Env or (if that's not
' writable) %LOCALAPPDATA%\FMIG_Env - this reads that pointer rather than
' guessing which one it used.

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
repoDir = fso.GetParentFolderName(scriptDir)
appScript = scriptDir & "\fmig_rat_app.py"
setupBat = repoDir & "\setup_environment.bat"
pointerFile = shell.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\FMIG\env_location.txt"

Function ResolvePythonw()
    ResolvePythonw = ""
    If fso.FileExists(pointerFile) Then
        Set f = fso.OpenTextFile(pointerFile, 1)
        If Not f.AtEndOfStream Then
            installRoot = Trim(f.ReadLine())
            candidate = installRoot & "\python\pythonw.exe"
            If fso.FileExists(candidate) Then
                ResolvePythonw = candidate
            End If
        End If
        f.Close
    End If
End Function

pythonw = ResolvePythonw()

If pythonw = "" Then
    If Not fso.FileExists(setupBat) Then
        MsgBox "The Python environment this app needs hasn't been set up yet, and " & _
               "setup_environment.bat wasn't found at:" & vbCrLf & setupBat, _
               vbCritical, "FMIG Rat Reconstruction"
        WScript.Quit 1
    End If

    answer = MsgBox( _
        "The Python environment this app needs hasn't been set up yet." & vbCrLf & vbCrLf & _
        "Run setup_environment.bat now? It downloads Python and the required packages " & _
        "(a few hundred MB) into a self-contained folder - this takes a few minutes and " & _
        "only needs to happen once.", _
        vbYesNo + vbQuestion, "FMIG Rat Reconstruction")

    If answer <> vbYes Then
        WScript.Quit 1
    End If

    ' Run setup and wait for it to finish (window style 1 = normal, so
    ' progress is visible; True = wait) before trying to launch the app.
    shell.CurrentDirectory = repoDir
    shell.Run """" & setupBat & """", 1, True

    pythonw = ResolvePythonw()
    If pythonw = "" Then
        MsgBox "Setup finished but the environment still isn't where it was expected. " & _
               "Check the setup_environment.bat window for errors, then try again.", _
               vbCritical, "FMIG Rat Reconstruction"
        WScript.Quit 1
    End If
End If

cmd = """" & pythonw & """ """ & appScript & """"
shell.CurrentDirectory = scriptDir
shell.Run cmd, 0, False
