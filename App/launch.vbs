' Silent launcher for the FMIG Rat Reconstruction app.
' Runs the GUI with pythonw.exe (no console window) using the mi-env
' conda environment - the same one the Automated-FMIG-Rat.ipynb notebook uses.

Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

pythonw = "C:\Users\milabs\.conda\envs\mi-env\pythonw.exe"
appScript = scriptDir & "\fmig_rat_app.py"

Set shell = CreateObject("WScript.Shell")

If Not fso.FileExists(pythonw) Then
    MsgBox "Could not find the mi-env Python environment at:" & vbCrLf & pythonw & vbCrLf & vbCrLf & _
           "Edit launch.vbs (and MI_ENV_PYTHON in fmig_rat_app.py) if the environment moved.", _
           vbCritical, "FMIG Rat Reconstruction"
    WScript.Quit 1
End If

cmd = """" & pythonw & """ """ & appScript & """"
shell.CurrentDirectory = scriptDir
shell.Run cmd, 0, False
