@echo off
setlocal enabledelayedexpansion
rem Reconstruct ALL phases of a MILabs U-CT gated scan with RTK (CUDA).
rem Double-click this file, or run from cmd/PowerShell:
rem   recon_phase.bat [root_or_phase_dir] [extra args for milabs_rtk_recon.py]

rem ============ EDIT THIS: path to the Optimized_Reconstruction folder =========
set "ROOT=D:\Data\PhNd7\2026-07-20_13h53\Optimized_Reconstruction-MI"
rem Extra options applied to every phase (see milabs_rtk_recon.py --help), e.g.
rem for a fast preview:  set "EXTRA_ARGS=--downsample 4 --vol-scale 4"
set "EXTRA_ARGS="
rem =============================================================================

rem The RTK/CUDA stack runs in the self-contained "RTK/recon" Python env that
rem setup_environment.bat builds (see that script's header comment - it lands
rem at C:\FMIG_Env\rtk_python, or under %LOCALAPPDATA%\FMIG_Env\rtk_python if
rem C:\ isn't writable). That script records which one it used in a pointer
rem file, read here instead of guessing.
set "PYTHON="
set "POINTER=%LOCALAPPDATA%\FMIG\env_location.txt"
if exist "%POINTER%" (
    set /p INSTALL_ROOT=<"%POINTER%"
    if exist "!INSTALL_ROOT!\rtk_python\python.exe" set "PYTHON=!INSTALL_ROOT!\rtk_python\python.exe"
)
if "%PYTHON%"=="" (
    echo error: the RTK/recon Python environment hasn't been set up yet. 1>&2
    echo Run setup_environment.bat in the repo root first, then try again. 1>&2
    goto :done_fail
)
rem milabs_rtk_recon.py is vendored into this repo under vendor\RTKRecon (a
rem runtime-only subset of the full RTKRecon project) - resolved relative to
rem this .bat file's own folder so it works regardless of where the repo sits.
set "SCRIPT=%~dp0vendor\RTKRecon\milabs_rtk_recon.py"

if "%~1"=="" (
    if not exist "%ROOT%\" (
        echo error: "%ROOT%" does not exist - edit ROOT at the top of this file 1>&2
        goto :done_fail
    )
    "%PYTHON%" "%SCRIPT%" "%ROOT%" %EXTRA_ARGS%
) else (
    "%PYTHON%" "%SCRIPT%" %*
)
set "EXITCODE=%ERRORLEVEL%"
goto :done

:done_fail
set "EXITCODE=1"
:done
rem Keep the window open only when launched by double-click (Explorer runs cmd /c).
echo %cmdcmdline% | find /i "/c" >nul && (echo. & pause)
exit /b %EXITCODE%
