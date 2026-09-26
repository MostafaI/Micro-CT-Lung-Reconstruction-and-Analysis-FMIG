@echo off
rem Watchdog for recon_server.py: restarts it whenever it exits, whether that's
rem a clean VRAM/phase-budget recycle or a crash. Runs hidden (no console) when
rem launched from the app; to stop the server for good, use Task Manager (or
rem the app's own controls) rather than closing a window, since there isn't one.
setlocal enabledelayedexpansion
rem The RTK/CUDA stack runs in the self-contained "RTK/recon" Python env that
rem setup_environment.bat (repo root) builds - see that script's header
rem comment. It writes where it put the env to a pointer file, read here
rem rather than guessing between C:\FMIG_Env and %LOCALAPPDATA%\FMIG_Env.
set "PYTHON="
set "POINTER=%LOCALAPPDATA%\FMIG\env_location.txt"
if exist "%POINTER%" (
    set /p INSTALL_ROOT=<"%POINTER%"
    if exist "!INSTALL_ROOT!\rtk_python\python.exe" set "PYTHON=!INSTALL_ROOT!\rtk_python\python.exe"
)
if "%PYTHON%"=="" (
    rem This runs hidden with no console (see header comment), so there's no
    rem one to prompt - just fail once into recon_server.log instead of
    rem looping forever retrying an environment that doesn't exist yet.
    echo [watchdog] the RTK/recon Python environment hasn't been set up yet - run setup_environment.bat in the repo root.
    exit /b 2
)
set "SCRIPT=%~dp0recon_server.py"

:loop
"%PYTHON%" -u "%SCRIPT%" %*
if %ERRORLEVEL% EQU 99 (
    echo [watchdog] stop requested - not restarting.
    exit /b 0
)
echo.
echo [watchdog] recon_server.py exited with code %ERRORLEVEL% - restarting in 2s...
rem "timeout" needs a real console and fails silently when run hidden; "ping"
rem doesn't, so it's the standard headless-safe sleep trick in batch scripts.
ping -n 3 127.0.0.1 >nul
goto loop
