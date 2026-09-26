@echo off
rem Double-click this to set up everything the FMIG pipeline needs: a
rem self-contained Python environment for the app/segmentation/analysis code,
rem and a separate one for GPU (RTK/CUDA) reconstruction. Safe to re-run -
rem it skips reinstalling if already set up, unless you pass -Force:
rem   setup_environment.bat -Force
rem
rem The real logic lives in setup_environment.ps1 (next to this file) -
rem PowerShell is much better suited than batch for downloading/extracting
rem archives and editing files, so this .bat is just a launcher for it.
setlocal
set "PS1=%~dp0setup_environment.ps1"

if not exist "%PS1%" (
    echo error: setup_environment.ps1 not found next to this script: %PS1% 1>&2
    goto :done_fail
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
set "EXITCODE=%ERRORLEVEL%"
goto :done

:done_fail
set "EXITCODE=1"
:done
rem Keep the window open only when launched by double-click (Explorer runs cmd /c).
echo %cmdcmdline% | find /i "/c" >nul && (echo. & pause)
exit /b %EXITCODE%
