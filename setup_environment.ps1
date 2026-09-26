<#
.SYNOPSIS
    Sets up a self-contained, portable Python environment for the FMIG
    Micro-CT Lung Reconstruction pipeline. Run via setup_environment.bat
    (double-click it) rather than calling this file directly.

.DESCRIPTION
    Creates TWO plain (non-conda) embeddable Python 3.10 installs, mirroring
    the two interpreters this pipeline has always needed:

      <install root>\python\      "general" env - numpy/scipy/torch/etc.
                                   used by App/fmig_rat_app.py and
                                   App/pipeline_driver.py (replaces the old
                                   personal "mi-env" conda environment).

      <install root>\rtk_python\  "RTK" env - itk + itk-rtk for CUDA cone-beam
                                   reconstruction, used by recon_phase.bat and
                                   vendor/RTKRecon/start_recon_server.bat
                                   (replaces the old personal system Python
                                   3.10 install). Kept separate from the
                                   general env because vendor/RTKRecon's own
                                   code comments document DLL conflicts when
                                   itk/RTK shares an environment with a conda
                                   install - see milabs_rtk_recon.py.

    <install root> is C:\FMIG_Env if that's writable, otherwise
    %LOCALAPPDATA%\FMIG_Env. Either way, the resolved path is written to a
    fixed pointer file at %LOCALAPPDATA%\FMIG\env_location.txt so the app,
    launch.vbs, recon_phase.bat, etc. can all find it without each having to
    re-implement the C:\ -> LOCALAPPDATA fallback.

    Does NOT install: a CUDA Toolkit/GPU driver (hardware-specific, multi-GB,
    needs matching hardware - this script only detects and reports whether
    one is present), or ANTs (a separate C++ toolkit, not a Python library -
    also only detected/reported).

.PARAMETER Force
    Reinstall from scratch even if a previous successful setup is detected.
#>
[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Continue"  # NOT "Stop": pip and other native tools write non-fatal
                                      # notices to stderr, and with ErrorActionPreference=Stop,
                                      # PowerShell turns *any* stderr line from a native command
                                      # into a terminating exception regardless of its exit code
                                      # (confirmed by testing). Native-command failures are instead
                                      # caught explicitly via $LASTEXITCODE in Invoke-Checked;
                                      # PowerShell cmdlets below pass -ErrorAction Stop individually
                                      # where they need to hard-fail.
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest is much faster without the progress UI

$PYTHON_VERSION = "3.10.11"
$EMBED_URL = "https://www.python.org/ftp/python/$PYTHON_VERSION/python-$PYTHON_VERSION-embed-amd64.zip"
$GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

# Packages for the general app env. torch is installed separately (CPU wheel,
# see Install-GeneralEnv) since it needs its own package index.
$GENERAL_PACKAGES = @(
    "pillow", "dicom2nifti", "imageio", "matplotlib", "nibabel", "numba",
    "numpy", "pandas", "plotly", "pydicom", "scipy", "scikit-learn",
    "seaborn", "tqdm"
)

# Packages for the RTK/recon env. itk + itk-rtk are installed separately
# (pinned to the versions this repo's own code comments document as tested -
# see vendor/RTKRecon/milabs_rtk_recon.py).
$RTK_PACKAGES = @("numpy", "pandas", "nibabel", "tifffile", "tqdm")
$ITK_VERSION = "5.4.6"
$ITK_RTK_VERSION = "2.7"

$PointerDir = Join-Path $env:LOCALAPPDATA "FMIG"
$PointerFile = Join-Path $PointerDir "env_location.txt"

function Write-Section($title) {
    Write-Host ""
    Write-Host "==== $title ====" -ForegroundColor Cyan
}

function Invoke-Checked {
    # Runs a native executable and throws if it exits non-zero - PowerShell
    # does not do this by default for native commands.
    #
    # NOTE: the parameter below is deliberately called $ArgList, not $Args -
    # $Args is PowerShell's reserved automatic variable for unbound
    # positional parameters, and a formal parameter with that exact name
    # silently fails to bind (confirmed: the caller's array never arrives,
    # so `& $Exe @Args` runs $Exe with zero arguments - e.g. "python.exe"
    # with no script, which drops into an interactive REPL instead of
    # actually running pip).
    #
    # Also: piping through Write-Host (not just letting it stream to the
    # success/output stream) matters beyond visibility - a PowerShell
    # function's return value is EVERY unsuppressed object written to the
    # pipeline during its execution, not just what follows `return`. Without
    # this, a caller like `$pythonExe = New-EmbeddablePythonEnv ...` that
    # calls Invoke-Checked internally gets back an array of [pip's stdout
    # lines..., the actual path], and `$pythonExe` silently becomes garbage
    # (confirmed: this broke the next stage, which tried to run pip's own
    # install log text as if it were an executable name).
    param([string]$Exe, [string[]]$ArgList, [string]$FailMessage)
    & $Exe @ArgList 2>&1 | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) {
        throw "$FailMessage (exit code $LASTEXITCODE)"
    }
}

function Get-InstallRoot {
    $preferred = "C:\FMIG_Env"
    try {
        New-Item -ItemType Directory -Path $preferred -Force -ErrorAction Stop | Out-Null
        $probe = Join-Path $preferred ".writetest"
        [IO.File]::WriteAllText($probe, "ok")
        Remove-Item $probe -Force
        return $preferred
    } catch {
        $fallback = Join-Path $env:LOCALAPPDATA "FMIG_Env"
        Write-Host "C:\FMIG_Env is not writable ($($_.Exception.Message)) - using $fallback instead." -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $fallback -Force | Out-Null
        return $fallback
    }
}

function Get-EmbeddablePythonZip($cacheDir) {
    $zipPath = Join-Path $cacheDir "python-$PYTHON_VERSION-embed-amd64.zip"
    if (-not (Test-Path $zipPath)) {
        Write-Host "Downloading Python $PYTHON_VERSION (embeddable, ~8 MB) ..."
        Invoke-WebRequest -Uri $EMBED_URL -OutFile $zipPath -ErrorAction Stop
    } else {
        Write-Host "Using cached Python download at $zipPath"
    }
    return $zipPath
}

function New-EmbeddablePythonEnv {
    # Extracts the embeddable Python zip into $EnvDir, enables site-packages
    # (disabled by default in embeddable distributions), and bootstraps pip.
    # Returns the path to python.exe.
    param([string]$EnvDir, [string]$Label, [string]$ZipPath, [string]$CacheDir)

    Write-Host "Setting up $Label environment at $EnvDir ..."
    if (Test-Path $EnvDir) { Remove-Item $EnvDir -Recurse -Force -ErrorAction Stop }
    New-Item -ItemType Directory -Path $EnvDir -ErrorAction Stop | Out-Null
    Expand-Archive -Path $ZipPath -DestinationPath $EnvDir -Force -ErrorAction Stop

    $pthFile = Get-ChildItem -Path $EnvDir -Filter "python3*._pth" | Select-Object -First 1
    if (-not $pthFile) { throw "Could not find python3*._pth in $EnvDir after extracting the embeddable zip" }
    (Get-Content $pthFile.FullName -ErrorAction Stop) -replace '^\s*#\s*(import site)\s*$', '$1' |
        Set-Content $pthFile.FullName -ErrorAction Stop

    $pythonExe = Join-Path $EnvDir "python.exe"
    $getPipPath = Join-Path $CacheDir "get-pip.py"
    if (-not (Test-Path $getPipPath)) {
        Write-Host "Downloading get-pip.py ..."
        Invoke-WebRequest -Uri $GET_PIP_URL -OutFile $getPipPath -ErrorAction Stop
    }
    Invoke-Checked -Exe $pythonExe -ArgList @($getPipPath, "--no-warn-script-location") `
        -FailMessage "pip bootstrap failed for $Label env"

    return $pythonExe
}

function Find-ExistingTkPython {
    # The embeddable Python distribution this script otherwise uses NEVER
    # includes tkinter - confirmed by testing: it's deliberately stripped from
    # that package (not something pip can install; tkinter isn't a PyPI
    # package). fmig_rat_app.py's desktop GUI needs it. This looks for an
    # already-installed Python 3.10.x (any patch version - CPython's ABI is
    # stable within a minor version, confirmed by testing: copying
    # _tkinter.pyd from a 3.10.2 install into a 3.10.11 embeddable env worked
    # fine) with tkinter, to copy Tcl/Tk from.
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310"),
        "C:\Python310",
        "C:\Program Files\Python310"
    )
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $base = (& py -3.10 -c "import sys; print(sys.base_prefix)" 2>$null | Out-String).Trim()
            if ($base) { $candidates = @($base) + $candidates }
        } catch {}
    }
    foreach ($dir in $candidates) {
        if ((Test-Path (Join-Path $dir "Lib\tkinter\__init__.py")) -and (Test-Path (Join-Path $dir "DLLs\_tkinter.pyd"))) {
            return $dir
        }
    }
    return $null
}

function Add-TkinterSupport {
    # Bolts tkinter onto an embeddable Python env, two strategies in order:
    #   1. Copy it from an existing compatible Python 3.10.x already on this
    #      machine (fast, no download, and works even where installers are
    #      blocked by local security policy - see strategy 2's comment).
    #   2. Silently run the official Python 3.10.11 installer into a SCRATCH
    #      folder just to harvest its Tcl/Tk files, then delete that scratch
    #      install (never used as the real env). Only tried if strategy 1
    #      finds nothing. Some machines block this outright via local
    #      security policy (confirmed by testing: exit code 1625,
    #      "This installation is forbidden by system policy" - happens even
    #      as an Administrators-group member, non-elevated, per-user
    #      install) - if so, this whole function just returns $false and the
    #      general env still works for everything except the desktop GUI.
    param([string]$EnvDir, [string]$CacheDir)

    $source = Find-ExistingTkPython
    $tempInstall = $null
    if (-not $source) {
        Write-Host "No existing Python 3.10.x with tkinter found - trying the official installer as a source (into a scratch folder, not installed permanently) ..."
        try {
            $tempInstall = Join-Path $env:TEMP "fmig_py310_tk_source"
            if (Test-Path $tempInstall) { Remove-Item $tempInstall -Recurse -Force }
            $installerPath = Join-Path $CacheDir "python-$PYTHON_VERSION-amd64.exe"
            if (-not (Test-Path $installerPath)) {
                Invoke-WebRequest -Uri "https://www.python.org/ftp/python/$PYTHON_VERSION/python-$PYTHON_VERSION-amd64.exe" -OutFile $installerPath -ErrorAction Stop
            }
            $p = Start-Process -FilePath $installerPath -ArgumentList @(
                "/quiet", "InstallAllUsers=0", "TargetDir=$tempInstall", "PrependPath=0",
                "Include_launcher=0", "Include_test=0", "AssociateFiles=0", "Shortcuts=0",
                "Include_tcltk=1", "Include_pip=0", "CompileAll=0"
            ) -Wait -PassThru
            if ($p.ExitCode -eq 0 -and (Test-Path (Join-Path $tempInstall "Lib\tkinter\__init__.py"))) {
                $source = $tempInstall
            } else {
                Write-Host "Installer-based source failed (exit code $($p.ExitCode))." -ForegroundColor Yellow
            }
        } catch {
            Write-Host "Installer-based source failed: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }

    if (-not $source) {
        Write-Host "Could not find or obtain tkinter from any source - the desktop GUI (App\fmig_rat_app.py) won't run in this environment. Everything else (pipeline_driver.py run directly, segmentation, analysis, registration) still works." -ForegroundColor Yellow
        return $false
    }

    Write-Host "Adding tkinter support from $source ..."
    # Placed in lib\site-packages (already on sys.path via the embeddable
    # env's python3*._pth), not Lib\tkinter (matching the full-installer
    # layout $source uses) - the embeddable env's own python3*._pth doesn't
    # put "Lib" on sys.path at all, confirmed by testing (a first attempt
    # copying to Lib\tkinter was silently invisible to `import tkinter`).
    # Likewise _tkinter.pyd/the Tcl DLLs go in $EnvDir's own root, not a DLLs
    # subfolder - the embeddable layout keeps its own .pyd files there directly.
    $tkPkgDir = Join-Path $EnvDir "lib\site-packages\tkinter"
    New-Item -ItemType Directory -Path $tkPkgDir -Force | Out-Null
    Copy-Item (Join-Path $source "Lib\tkinter\*") $tkPkgDir -Recurse -Force
    Copy-Item (Join-Path $source "DLLs\_tkinter.pyd") $EnvDir -Force
    Copy-Item (Join-Path $source "DLLs\tcl86t.dll") $EnvDir -Force
    Copy-Item (Join-Path $source "DLLs\tk86t.dll") $EnvDir -Force
    $tclDest = Join-Path $EnvDir "tcl"
    if (Test-Path $tclDest) { Remove-Item $tclDest -Recurse -Force }
    Copy-Item (Join-Path $source "tcl") $tclDest -Recurse -Force

    if ($tempInstall -and (Test-Path $tempInstall)) {
        Remove-Item $tempInstall -Recurse -Force -ErrorAction SilentlyContinue
    }

    $out = & (Join-Path $EnvDir "python.exe") -c "import tkinter; print('TK_OK ' + str(tkinter.TkVersion))" 2>&1
    $outText = $out | Out-String
    if ($outText -match "TK_OK") {
        return $true
    }
    Write-Host "tkinter was copied but failed to import:`n$outText" -ForegroundColor Yellow
    return $false
}

function Install-GeneralEnv($pythonExe) {
    Write-Section "Installing general app packages ($($GENERAL_PACKAGES.Count) packages + torch)"
    Invoke-Checked -Exe $pythonExe -ArgList (@("-m", "pip", "install", "--no-warn-script-location") + $GENERAL_PACKAGES) `
        -FailMessage "Installing general app packages failed"

    # CPU-only torch: keeps this script portable to machines with no NVIDIA
    # GPU or no matching CUDA Toolkit. The app already falls back to CPU
    # automatically (torch.cuda.is_available() checks in segment_from_projections.py),
    # and the only place torch is used is a single small U-Net forward pass per
    # image, which is fine on CPU. If this machine has a supported NVIDIA GPU
    # and you want faster segmentation, you can later run:
    #   <this env>\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
    Write-Host "Installing torch (CPU build) ..."
    Invoke-Checked -Exe $pythonExe -ArgList @("-m", "pip", "install", "--no-warn-script-location", "torch", "--index-url", "https://download.pytorch.org/whl/cpu") `
        -FailMessage "Installing torch failed"
}

function Get-CudaWheelSuffix($ToolkitVersions) {
    # Maps a detected CUDA Toolkit version dir name (e.g. "v12.1") to the
    # itk-rtk-cudaXXX / itk-cudacommon-cudaXXX PyPI package suffix - RTK
    # Consortium only publishes wheels for specific CUDA minor versions.
    # Checked directly against PyPI (2026-09): cuda116, cuda121, cuda124
    # exist; other versions 404. No API to query this at install time, so
    # it's a fixed list - update it here if RTK Consortium adds more.
    $known = @("124", "121", "116")
    foreach ($v in $ToolkitVersions) {
        if ($v -match 'v(\d+)\.(\d+)') {
            $suffix = "$($Matches[1])$($Matches[2])"
            if ($known -contains $suffix) { return $suffix }
        }
    }
    return $null
}

function Install-RtkEnv($pythonExe, $cudaToolkitVersions) {
    Write-Section "Installing RTK/recon packages"
    Invoke-Checked -Exe $pythonExe -ArgList (@("-m", "pip", "install", "--no-warn-script-location") + $RTK_PACKAGES) `
        -FailMessage "Installing RTK env base packages failed"

    # The plain "itk-rtk" PyPI package is CPU-only - confirmed by testing:
    # hasattr(rtk, 'CudaFDKConeBeamReconstructionFilter') is False with it,
    # even on a machine with a working CUDA Toolkit + GPU. CUDA acceleration
    # needs the matching "itk-rtk-cudaXXX" variant instead (which pulls in
    # itk-cudacommon-cudaXXX automatically) - this repo's own code comment
    # ("pip itk-rtk 2.7 / itk 5.4.6") doesn't distinguish the two, so prefer
    # the CUDA variant whenever this machine's CUDA Toolkit version has one
    # published, and fall back to CPU-only itk-rtk otherwise.
    $cudaSuffix = Get-CudaWheelSuffix $cudaToolkitVersions
    $installedCuda = $false
    if ($cudaSuffix) {
        Write-Host "Installing itk==$ITK_VERSION itk-rtk-cuda$cudaSuffix (CUDA-accelerated, matches this machine's CUDA Toolkit) ..."
        & $pythonExe -m pip install --no-warn-script-location "itk==$ITK_VERSION" "itk-rtk-cuda$cudaSuffix"
        if ($LASTEXITCODE -eq 0) {
            $installedCuda = $true
        } else {
            Write-Host "itk-rtk-cuda$cudaSuffix install failed - falling back to the CPU-only itk-rtk package." -ForegroundColor Yellow
        }
    } else {
        Write-Host "No published itk-rtk-cudaXXX wheel matches this machine's CUDA Toolkit - installing CPU-only itk-rtk." -ForegroundColor Yellow
    }

    if (-not $installedCuda) {
        Write-Host "Installing itk==$ITK_VERSION itk-rtk==$ITK_RTK_VERSION (CPU-only) ..."
        & $pythonExe -m pip install --no-warn-script-location "itk==$ITK_VERSION" "itk-rtk==$ITK_RTK_VERSION"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Pinned itk/itk-rtk install failed - retrying with unpinned latest versions ..." -ForegroundColor Yellow
            Invoke-Checked -Exe $pythonExe -ArgList @("-m", "pip", "install", "--no-warn-script-location", "itk", "itk-rtk") `
                -FailMessage "Installing itk/itk-rtk failed even unpinned"
        }
    }
}

function Test-GeneralEnv($pythonExe) {
    # tkinter is checked but its absence doesn't fail verification - only
    # fmig_rat_app.py's desktop GUI needs it; pipeline_driver.py (run
    # directly) and everything else work without it. See Add-TkinterSupport.
    $code = @"
import numpy, scipy, pandas, matplotlib, nibabel, pydicom, dicom2nifti, sklearn, seaborn, plotly, numba, tqdm, torch
from PIL import Image
try:
    import tkinter
    tk_status = 'tkinter=' + str(tkinter.TkVersion)
except Exception as e:
    tk_status = 'tkinter=MISSING(' + str(e) + ')'
print('GENERAL_ENV_OK torch=' + torch.__version__ + ' cuda_available=' + str(torch.cuda.is_available()) + ' ' + tk_status)
"@
    # $out from `2>&1` is an array (one element per line) whenever the command
    # prints more than one line (e.g. a deprecation notice alongside the OK
    # marker) - and -match/-notmatch against an ARRAY returns the subset of
    # elements that (don't) match, not a boolean, so testing it directly in
    # `if` would treat "OK line + one warning line" as a failure. Join to a
    # single string first so -notmatch is the scalar boolean test intended.
    $out = & $pythonExe -c $code 2>&1
    $outText = $out | Out-String
    if ($LASTEXITCODE -ne 0 -or ($outText -notmatch "GENERAL_ENV_OK")) {
        throw "General env verification failed:`n$outText"
    }
    return $outText
}

function Test-RtkEnv($pythonExe) {
    # Import order matters: this repo's own code (milabs_rtk_recon.py) documents
    # that itk must be imported - and a first RTK object constructed - before
    # pandas/nibabel, or a conflicting DLL causes an access violation on
    # Windows (pip itk-rtk 2.7 / itk 5.4.6). The CUDA toolkit bin dir must also
    # be on the DLL search path before itk loads, for the same reason
    # milabs_rtk_recon.py registers it at import time.
    $code = @"
import os, glob, sys
for d in sorted(glob.glob(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin')):
    os.add_dll_directory(d)
import itk
from itk import RTK as rtk
f3 = itk.Image[itk.F, 3]
rtk.ConstantImageSource[f3].New()
rtk.FDKConeBeamReconstructionFilter[f3].New()
have_cuda = hasattr(rtk, 'CudaFDKConeBeamReconstructionFilter')
import numpy, pandas, nibabel, tifffile, tqdm
print('RTK_ENV_OK itk=' + itk.Version.GetITKVersion() + ' cuda_filter_available=' + str(have_cuda))
"@
    # See the identical Out-String note in Test-GeneralEnv above.
    $out = & $pythonExe -c $code 2>&1
    $outText = $out | Out-String
    if ($LASTEXITCODE -ne 0 -or ($outText -notmatch "RTK_ENV_OK")) {
        throw "RTK env verification failed:`n$outText"
    }
    return $outText
}

function Test-CudaToolkit {
    $dirs = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -ErrorAction SilentlyContinue |
        Where-Object { $_.PSIsContainer }
    $versions = @($dirs | ForEach-Object { $_.Name })
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    return [PSCustomObject]@{
        ToolkitFound = [bool]$dirs
        Versions = $versions
        NvidiaSmiFound = [bool]$nvidiaSmi
    }
}

function Test-Ants {
    # Matches register_phases.py's DEFAULT_ANTSPATH.
    $antsExe = Join-Path $env:USERPROFILE "ants\ants-2.6.5\bin\antsRegistration.exe"
    return Test-Path $antsExe
}

# ----------------------------------------------------------------------------

Write-Host "FMIG environment setup" -ForegroundColor Green
Write-Host "======================="

$InstallRoot = Get-InstallRoot
$MarkerFile = Join-Path $InstallRoot ".setup_complete"
New-Item -ItemType Directory -Path $PointerDir -Force | Out-Null

if ((Test-Path $MarkerFile) -and (-not $Force)) {
    Write-Host "Already set up at $InstallRoot (marker file found)." -ForegroundColor Green
    Write-Host "Re-run with -Force to reinstall from scratch: setup_environment.bat -Force"
    Set-Content -Path $PointerFile -Value $InstallRoot -NoNewline
    exit 0
}

Write-Host "Install root: $InstallRoot"
$LogFile = Join-Path $InstallRoot "setup.log"
Start-Transcript -Path $LogFile -Force | Out-Null

$overallOk = $true
$rtkOk = $true

try {
    $cacheDir = Join-Path $InstallRoot "_downloads"
    New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
    $zipPath = Get-EmbeddablePythonZip $cacheDir

    Write-Section "General app environment"
    $appEnvDir = Join-Path $InstallRoot "python"
    $appPython = New-EmbeddablePythonEnv -EnvDir $appEnvDir -Label "general app" -ZipPath $zipPath -CacheDir $cacheDir
    Install-GeneralEnv $appPython
    $guiAvailable = Add-TkinterSupport -EnvDir $appEnvDir -CacheDir $cacheDir
    $generalResult = Test-GeneralEnv $appPython
    Write-Host $generalResult -ForegroundColor Green
    if ($generalResult -match "tkinter=MISSING") {
        $guiAvailable = $false
    }
    if (-not $guiAvailable) {
        Write-Host "Desktop GUI (App\fmig_rat_app.py) will NOT run in this environment (see above). Everything else works - run App\pipeline_driver.py directly instead (see its own --help)." -ForegroundColor Yellow
    }

    Write-Section "Hardware/toolkit checks (informational - not installed by this script)"
    $cuda = Test-CudaToolkit
    if ($cuda.ToolkitFound) {
        Write-Host "CUDA Toolkit found: $($cuda.Versions -join ', ')" -ForegroundColor Green
    } else {
        Write-Host "CUDA Toolkit NOT found under 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'. GPU reconstruction needs it installed separately (see NVIDIA's site)." -ForegroundColor Yellow
    }
    Write-Host "nvidia-smi (GPU driver) found: $($cuda.NvidiaSmiFound)"

    if (Test-Ants) {
        Write-Host "ANTs found at the expected path (see register_phases.py's DEFAULT_ANTSPATH)." -ForegroundColor Green
    } else {
        Write-Host "ANTs NOT found at the expected path (register_phases.py's DEFAULT_ANTSPATH). Phase registration needs it installed separately (not a Python package - not handled by this script)." -ForegroundColor Yellow
    }

    Write-Section "RTK/recon environment"
    $rtkCudaAccelerated = $false
    try {
        $rtkPython = New-EmbeddablePythonEnv -EnvDir (Join-Path $InstallRoot "rtk_python") -Label "RTK/recon" -ZipPath $zipPath -CacheDir $cacheDir
        Install-RtkEnv $rtkPython $cuda.Versions
        $rtkResult = Test-RtkEnv $rtkPython
        Write-Host $rtkResult -ForegroundColor Green
        $rtkCudaAccelerated = $rtkResult -match "cuda_filter_available=True"
        if (-not $rtkCudaAccelerated) {
            Write-Host "RTK env works but has NO CUDA acceleration (no matching itk-rtk-cudaXXX wheel for this machine's CUDA Toolkit) - reconstruction will run on CPU, which is much slower." -ForegroundColor Yellow
        }
    } catch {
        $rtkOk = $false
        Write-Host "RTK/recon environment setup failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "The general app (segmentation, analysis, registration) will still work." -ForegroundColor Yellow
        Write-Host "Reconstruction will not, until this is fixed (re-run setup_environment.bat -Force)." -ForegroundColor Yellow
    }

    Set-Content -Path $PointerFile -Value $InstallRoot -NoNewline

    if ($overallOk) {
        $marker = @{
            installed_at = (Get-Date).ToString("o")
            install_root = $InstallRoot
            python_version = $PYTHON_VERSION
            general_env_ok = $true
            gui_available = $guiAvailable
            rtk_env_ok = $rtkOk
            rtk_cuda_accelerated = $rtkCudaAccelerated
        } | ConvertTo-Json
        Set-Content -Path $MarkerFile -Value $marker
    }

    Write-Section "Done"
    $guiNote = if ($guiAvailable) { "GUI available" } else { "GUI NOT available - see warning above" }
    Write-Host "General app environment: $InstallRoot\python ($guiNote)" -ForegroundColor Green
    if ($rtkOk) {
        $accelNote = if ($rtkCudaAccelerated) { "CUDA-accelerated" } else { "CPU-only, no GPU acceleration" }
        Write-Host "RTK/recon environment:   $InstallRoot\rtk_python ($accelNote)" -ForegroundColor Green
    } else {
        Write-Host "RTK/recon environment:   FAILED (see above / $LogFile)" -ForegroundColor Red
    }
    Write-Host "Pointer file: $PointerFile -> $InstallRoot"
    Write-Host ""
    Write-Host "You can now launch the app: App\launch.vbs (or App\fmig_rat_app.py directly)."
} catch {
    Write-Host ""
    Write-Host "Setup FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Full log: $LogFile"
    Stop-Transcript | Out-Null
    exit 1
}

Stop-Transcript | Out-Null
exit 0
