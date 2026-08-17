@echo off
setlocal
rem Reconstruct ALL phases of a MILabs U-CT gated scan with RTK (CUDA).
rem Double-click this file, or run from cmd/PowerShell:
rem   recon_phase.bat [root_or_phase_dir] [extra args for milabs_rtk_recon.py]

rem ============ EDIT THIS: path to the Optimized_Reconstruction folder =========
set "ROOT=D:\Data\PhNd7\2026-07-20_13h53\Optimized_Reconstruction-MI"
rem Extra options applied to every phase (see milabs_rtk_recon.py --help), e.g.
rem for a fast preview:  set "EXTRA_ARGS=--downsample 4 --vol-scale 4"
set "EXTRA_ARGS="
rem =============================================================================

rem The RTK/CUDA stack lives in the system Python 3.10 - NOT in any conda env.
set "PYTHON=C:\Users\milabs\AppData\Local\Programs\Python\Python310\python.exe"
set "SCRIPT=C:\Users\milabs\Desktop\Pipeline\RTKRecon\milabs_rtk_recon.py"

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
