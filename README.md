# Micro-CT-Lung-Reconstruction-and-Analysis-FMIG

A pipeline for reconstructing and analyzing preclinical micro-CT images for lung disorders, currently used in the Functional and Metabolic Imaging Group (FMIG) at the University of Pennsylvania.

It takes raw or flat-field-corrected projections from a MILabs U-CT scanner, bins them into 16 respiratory phases, reconstructs each phase on the GPU with RTK (FDK cone-beam), segments the lungs, registers the breathing phases with ANTs, and produces functional lung maps, diaphragm motion, lung volumes, and regional ventilation-timing maps.

Developed by PhD candidate **Mostafa Ismail**, 2021–2026.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## Contents

- [Pipeline overview](#pipeline-overview)
- [Software requirements](#software-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Sample input](#sample-input)
- [Sample output](#sample-output)
- [Reproducibility](#reproducibility)
- [Repository layout](#repository-layout)
- [License](#license)

---

## Pipeline overview

The pipeline runs as a sequence of steps. Each step can be run on its own, and steps that already have output are skipped rather than overwritten.

| Step | What it does | Needs |
|---|---|---|
| `recon` | Corrects projections (flat-field + dead pixels), finds the diaphragm with a U-Net, extracts the breathing signal, bins projections into 16 phases, reconstructs every phase with RTK | Raw scan |
| `segment` | Segments the lungs in every phase volume | `recon` |
| `segment_lr` | Splits the lung masks into left and right lungs (automatic, or from a manual override drawn in the app) | `segment` |
| `register` | Registers end-expiration (EE) to end-inspiration (EI) with ANTs and extracts deformation fields | `segment` |
| `register_all_phases` | Registers every phase onto R0 (slow; needed for `time_maps`) | `segment` |
| `analysis` | Functional maps: FRC, TLC, tidal volume (TV), fractional ventilation (FV), Jacobian (J), Δblood | `segment` (+ `register` for TV/FV/J) |
| `diaphragm` | Diaphragm descent in mm (whole, left, right) with 2D/3D visualizations | `register` |
| `volume_analysis` | FRC/TLC/TV in mL per lung, written to a per-rat CSV | `segment` |
| `time_maps` | Per-voxel EI time, arrival time and expiratory time constant (tau) | `register_all_phases` |
| `register_to_baseline` | Registers every session of a rat to its first (baseline) session | Group mode only |

---

## Software requirements

### Operating system and hardware

| Item | Requirement |
|---|---|
| OS | Windows 10/11 or Windows Server 2019 (64-bit). The launchers are `.bat`, `.ps1` and `.vbs` files. |
| GPU | NVIDIA GPU with CUDA Toolkit **11.6, 12.1, or 12.4** for GPU reconstruction. Without one, RTK falls back to CPU (tens of minutes per scan instead of about 80 s). |
| RAM | 32 GB or more recommended. A full scan is 360 projections of 972 × 768 pixels per phase. |
| Disk | About 10–20 GB free per session for intermediate files. Registration outputs are pruned automatically. |

Development and testing were done on an NVIDIA Quadro P4000 with CUDA 12.1.

### External tools (installed separately)

| Tool | Version | Used by | Notes |
|---|---|---|---|
| [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) | 11.6 / 12.1 / 12.4 | GPU reconstruction | Install to the default `C:\Program Files\NVIDIA GPU Computing Toolkit` location |
| [ANTs](https://github.com/ANTsX/ANTs/releases) | 2.6.5 (native Windows build) | Phase registration | Expected at `%USERPROFILE%\ants\ants-2.6.5\bin` (see `DEFAULT_ANTSPATH` in `register_phases.py`) |
| [Git for Windows](https://git-scm.com/download/win) | any recent | Runs `yi2.sh` (the ANTs registration script) through Git Bash | WSL's `bash.exe` is not used |

### Python environments

Two separate, self-contained Python **3.10.11** environments are used. They are kept apart because ITK/RTK has DLL conflicts when it shares an environment with the scientific stack.

| Environment | Main packages | Used by |
|---|---|---|
| General (`<install root>\python`) | numpy, scipy, pandas, matplotlib, nibabel, pydicom, dicom2nifti, scikit-learn, seaborn, plotly, numba, tifffile, imageio, pillow, tqdm, torch (CPU build) | The app, `pipeline_driver.py`, segmentation, registration wrappers, analysis |
| RTK (`<install root>\rtk_python`) | itk 5.4.6, itk-rtk 2.7 (or the matching `itk-rtk-cudaXXX` build), numpy, pandas, nibabel, tifffile, tqdm | `recon_phase.bat` and the RTK reconstruction server |

Both are created automatically by the setup script below. Conda is not needed.

---

## Installation

1. **Install the external tools** listed above: CUDA Toolkit (optional but strongly recommended), ANTs 2.6.5, and Git for Windows.

2. **Clone the repository.**

   ```bat
   git clone https://github.com/MostafaI/Micro-CT-Lung-Reconstruction-and-Analysis-FMIG.git
   cd Micro-CT-Lung-Reconstruction-and-Analysis-FMIG
   ```

3. **Build the Python environments.** Double-click `setup_environment.bat`, or run it from a terminal:

   ```bat
   setup_environment.bat
   ```

   The script downloads embeddable Python 3.10.11, installs both environments, detects your CUDA Toolkit and picks the matching RTK wheel, then runs import tests and reports whether CUDA and ANTs were found. The environments go to `C:\FMIG_Env`, or to `%LOCALAPPDATA%\FMIG_Env` when `C:\` is not writable. The chosen location is recorded in `%LOCALAPPDATA%\FMIG\env_location.txt`, which every launcher reads.

   Re-running it is safe. Add `-Force` to rebuild from scratch:

   ```bat
   setup_environment.bat -Force
   ```

4. **Optional: enable GPU segmentation.** The general environment installs CPU-only PyTorch, which is enough for the small U-Net used here. For a CUDA build:

   ```bat
   C:\FMIG_Env\python\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
   ```

5. **Launch the app.** Double-click `App\launch.vbs`. It starts the GUI with no console window.

---

## Usage

### Desktop app

`App\launch.vbs` opens the FMIG Rat Reconstruction app. It offers three modes:

- **Single dataset**: run selected steps on one scan session.
- **Group**: run selected steps on every session of one or more rats.
- **Automated**: run selected steps on every session under a data root dated on or after a start date.

Progress, figures, and logs appear in the app as each step finishes. Errors are also written to `%LOCALAPPDATA%\FMIG\app_errors.log`.

### Command line

The same steps can be run headlessly with the general environment's Python:

```bat
set PY=C:\FMIG_Env\python\python.exe

rem One session, all steps
%PY% App\pipeline_driver.py "D:\Data\Rat01\2026-08-06_10h47"

rem One session, selected steps
%PY% App\pipeline_driver.py "D:\Data\Rat01\2026-08-06_10h47" --steps recon,segment,analysis

rem Every session of two rats
%PY% App\pipeline_driver.py --rats "D:\Data\Rat01" --rats "D:\Data\Rat02" --steps diaphragm

rem Every session since a date, skipping steps already done
%PY% App\pipeline_driver.py --data-root "D:\Data" --since 2026-10-01 --steps recon,segment --skip-done
```

Other options:

| Option | Default | Meaning |
|---|---|---|
| `--steps` | all | Comma-separated subset of the steps in the overview table |
| `--threshold` | 0.5 | Projection segmentation threshold used to extract the breathing signal |
| `--use-fallback-timing` | off | Use a known-good timing template when a scan's own per-projection timestamps are unusable. Output then goes to a separate `..._fallback` folder |

### Reconstruction only

To reconstruct an already-binned scan directly with RTK, edit `ROOT` at the top of `recon_phase.bat` and double-click it, or pass the folder:

```bat
recon_phase.bat "D:\Data\Rat01\2026-08-06_10h47\Optimized_Reconstruction-MI"
recon_phase.bat "D:\Data\Rat01\2026-08-06_10h47\Optimized_Reconstruction-MI\Phase_0" --downsample 4 --vol-scale 4
```

The second form is a fast, low-resolution preview of a single phase. Starting `vendor\RTKRecon\start_recon_server.bat` first keeps ITK/RTK loaded between runs and saves about 40 s per reconstruction. See [`vendor/RTKRecon/README.md`](vendor/RTKRecon/README.md) for the geometry and processing chain.

### Notebook

`Automated-FMIG-Rat.ipynb` contains the original interactive version of the pipeline. `App/pipeline_driver.py` reproduces it step for step and is the maintained entry point.

---

## Sample input

Data are organised as one folder per rat, with one subfolder per scan session named `YYYY-MM-DD_HHhMM`. Each session is a MILabs U-CT gated acquisition:

```
D:\Data\
└── Rat01\                                   rat folder
    ├── 2026-08-06_10h47\                    session folder
    │   ├── ct-data\
    │   │   ├── Chunk*.tif                   raw projections (used if corr\ is absent)
    │   │   ├── calibration\
    │   │   │   ├── geopar.cfg               scanner geometry (sdd, sid, offsets, angles)
    │   │   │   └── ...                      dark/white-field calibration images
    │   │   ├── corr\                        flat-field-corrected projections
    │   │   │   └── proj_000_0_000NNNNN.tif  (from MILabs software, or built by the pipeline)
    │   │   └── proj_000_0_log.csv           per-projection gantry angle and timing
    │   └── *.mCT                            MILabs scan file (HU calibration, beam hardening)
    └── 2026-08-13_11h02\
        └── ...
```

Either `ct-data\corr\` or the raw `Chunk*.tif` files plus `calibration\` must be present. When `corr\` is missing it is built from the raw chunks.

The pipeline was developed on free-breathing, retrospectively gated rat scans acquired at 55 kV, 37 mA, and 20 ms exposure, reconstructed into 16 respiratory phases. `sample_times_32p_55kv_37ma_20ms.npy` is the reference timing template for that protocol.

The imaging data are not distributed with this repository.

---

## Sample output

Per-session output goes to `<session>\Optimized_Reconstruction-MI\`, or to `..._fallback` when the fallback timing was used:

```
<session>\
├── MI_Rat_window_result.png             diaphragm detection on the projections
├── clustering_result.png                respiratory phase binning
├── fmig_code_version.json               code version and settings for every step run
└── Optimized_Reconstruction-MI\
    ├── Phase_0\ ... Phase_15\           binned projections per phase
    └── Results\
        ├── CT_<scan>_R0.nii.gz ... _R15.nii.gz          reconstructed phases (int16 HU)
        ├── CT_<scan>_R0_m.nii.gz ... _R15_m.nii.gz      lung masks
        ├── CT_..._R0_mLR.nii.gz, CT_..._R7_mLR.nii.gz   left/right labels (1 = left, 2 = right)
        ├── images\video.gif, images_cropped\video.gif   cine loops through the breathing cycle
        ├── raw.gif, mask.gif
        ├── pipeline_config.json                         detected EI/EE phases and settings
        ├── Maps\
        │   ├── maps.png                                 combined functional maps
        │   ├── FRC.png, TLC.png, TV.png, FV.png, J.png, delta_blood.png (+ _smoothed)
        │   ├── maps_panel_{EI,EE}_{coronal,sagittal,axial}.jpg
        │   └── time_maps_summary.png / .json, maps_panel_time_*.jpg
        └── Registered\R7_to_R0\
            ├── *Forward.nii.gz, *_TotalWarp.nii.gz      ANTs deformation fields
            ├── diaphragm_analysis.txt                   diaphragm descent (mm), whole/left/right
            ├── diaphragm_descent{,_left,_right}.html    interactive 3D plots
            └── diaphragm_2D_visualization_*.jpg
```

Per-rat output goes to `<rat>\Analysis\`:

| File | Content |
|---|---|
| `volume_analysis.csv` | One row per session: FRC, TLC, and TV in mL for left, right, and total lung |
| Left/right split panels | One image per session for checking the lung split |
| Coronal maps panels | EI and EE functional maps across all sessions of the rat |

---

## Reproducibility

### Version tracking

The pipeline version lives in the `VERSION` file (currently `v1.0`). Bump it whenever a change affects results. The app shows it in its title bar and at the top of every run log.

Every time a step finishes, the pipeline records which version produced it in `fmig_code_version.json` inside that session's folder. Group-level registration writes the same file into the rat's folder. The file holds:

- `steps`: the latest entry for each step, showing which version made that step's current output.
- `history`: every entry, oldest first.

Each entry has the version, git commit, whether there were uncommitted changes, the time, the machine, and the step's settings.

### Guidelines for reproducible results

- **Pin the environment.** Build environments only with `setup_environment.bat`. It pins Python 3.10.11, itk 5.4.6, and itk-rtk 2.7. To capture the exact package set of a working install, run:

  ```bat
  C:\FMIG_Env\python\python.exe -m pip freeze > requirements-general.lock.txt
  C:\FMIG_Env\rtk_python\python.exe -m pip freeze > requirements-rtk.lock.txt
  ```

- **Run from a clean commit.** `fmig_code_version.json` flags uncommitted changes. For results that go into a paper, commit first and quote the commit hash and `VERSION`.
- **Regenerate deliberately.** Steps skip sessions that already have output and never overwrite it. To regenerate a step after a code change, move or delete its output first, then rerun. `fmig_code_version.json` keeps older and newer results distinguishable.
- **Keep the model checkpoint fixed.** Diaphragm detection uses the U-Net checkpoint `vendor/Rat-lung-segmentation-from-xray/checkpoints/best_model_1.pt`, which is versioned in this repository.
- **Report the reconstruction device.** CUDA and CPU FDK give equivalent but not bit-identical volumes. `setup_environment.bat` reports which RTK build was installed.
- **Watch for fallback timing.** Sessions reconstructed with `--use-fallback-timing` are written to a separate `_fallback` folder so they are never mixed with normally timed sessions.
- **Check phase ordering on old data.** Phase labels are time-sorted from 2026-09-25 onwards. `Phase_N` folders produced before that date may be ordered differently and should be re-binned before comparing across sessions.
- **Adjust machine-specific paths.** A few defaults point to the development machine: the ANTs install path in `register_phases.py`, the example data root in `Analysis.py` and `recon_phase.bat`, and a fallback U-Net checkpoint path in `App/pipeline_driver.py`. Change them if your layout differs.

### Validation

The RTK reconstruction is matched to the MILabs vendor reconstruction on the same voxel grid, HU scale, filtering, and NIfTI header. Against a vendor volume it reaches a correlation of r ≈ 0.99 with no residual voxel shift. Details are in [`vendor/RTKRecon/README.md`](vendor/RTKRecon/README.md).

---

## Repository layout

| Path | Purpose |
|---|---|
| `App/fmig_rat_app.py` | Desktop GUI |
| `App/pipeline_driver.py` | Headless pipeline: all steps and the command-line interface |
| `App/launch.vbs` | Silent launcher for the GUI |
| `raw_correction.py` | Flat-field and dead-pixel correction of raw projections |
| `recon_functions.py`, `recon_functions_all.py` | Breathing-signal extraction, phase binning, MILabs folder structure |
| `segment_from_projections.py` | U-Net lung and diaphragm detection on projections |
| `register_phases.py`, `yi2.sh` | ANTs phase registration |
| `Analysis.py`, `utils.py` | Segmentation, functional maps, plotting helpers |
| `fmig_version.py`, `VERSION` | Code-version tracking |
| `setup_environment.bat`, `setup_environment.ps1` | Environment installer |
| `recon_phase.bat` | Stand-alone RTK reconstruction launcher |
| `vendor/RTKRecon/` | RTK reconstruction code and warm reconstruction server |
| `vendor/Rat-lung-segmentation-from-xray/` | U-Net model and checkpoint |
| `Automated-FMIG-Rat.ipynb` | Original interactive notebook |

---

## License

Copyright © 2021–2026 Mostafa Ismail, Functional and Metabolic Imaging Group, University of Pennsylvania.

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0). See [`LICENSE`](LICENSE) for the full text.

You may share and adapt this work for non-commercial purposes, provided you give appropriate credit and distribute your contributions under the same license. For commercial use, contact the author.

Third-party components keep their own licenses, including ITK and RTK (Apache 2.0), ANTs (Apache 2.0), and PyTorch (BSD-style).
