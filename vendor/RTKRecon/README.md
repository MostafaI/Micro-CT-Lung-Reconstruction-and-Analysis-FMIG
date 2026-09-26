# RTKRecon — MILabs U-CT reconstruction with RTK

Learning materials: `CT_Reconstruction_Tutorial.ipynb` (full pipeline, Python)
and `filtering_tutorial.m` (the filtering step in depth, MATLAB, base toolboxes
only — run section-by-section).

FDK cone-beam reconstruction of MILabs U-CT corrected projections, matched to
the vendor reconstruction (same voxel grid, HU scale, filtering and NIfTI
header).

## Usage

```
py milabs_rtk_recon.py "D:\Data\...\Optimized_Reconstruction"          # all phases
py milabs_rtk_recon.py "D:\Data\...\Optimized_Reconstruction\Phase_0"  # one phase
```

Or double-click `..\recon_phase.sh` (edit its `ROOT` variable once). All phases
run in one process because the first RTK template instantiation loads the whole
ITK DLL chain (~40 s, once per process — the "loading ITK/RTK libraries"
message).

Outputs into `<phase>/Results/`:

- `CT_<scan>_rtk.nii` — int16 HU volume on the same grid/header as the MILabs nii
- `CT_<scan>_rtk_geometry.xml` — RTK ThreeDCircularProjectionGeometry
- `CT_<scan>_rtk_geometry_matrices_mm.npy` — 360 × 3 × 4 projection matrices,
  homogeneous world (mm) → detector (u,v) in mm
- `CT_<scan>_rtk_geometry_matrices_pix.npy` — same but → detector pixel indices
- `CT_<scan>_rtk_geometry_matrices.txt` — human-readable dump of both

Preview run (≈16× faster): `--downsample 4 --vol-scale 4`.
Compare with the vendor image:

```
py compare_to_milabs.py <rtk.nii> <milabs.nii>
```

## Inputs and parameter sources

| Source | Parameters |
|---|---|
| `Results/CT_*.log` (MILabs recon dump, preferred when present) | everything below, plus `origin`/`niiorigin`, `i0`, `hann`, `gaussblr`, `wpc`, `hu`, grid |
| `ct-data/calibration/geopar.cfg` | `sdd, sid, proj_iso_x/y, source_x/y, in_angle, first_angle` |
| `ct-data/proj_000_0_log.csv` | per-projection gantry angle (3rd column) + `first_angle` |
| `*.mCT` | fallback for `HU_calibration`, `BeamHardening` |
| `ct-data/corr/*.tif` | corrected projections; detector pixel size from TIFF resolution tags (149.6 µm) |

## Processing chain (mirrors the vendor pipeline)

1. line integrals `p = ln(i0 / I)` with `i0 = 60000`
2. water precorrection `p ← wpc[0] + wpc[1]·p + wpc[2]·p²` (beam hardening)
3. RTK FDK, Hann-windowed ramp (`hann=1`), geometry from geopar mapped
   directly onto `AddProjection(sid, sdd, angle, proj_iso_x, proj_iso_y, 0,
   in_angle, source_x, source_y)`; projection stack origin at detector pixel 0
4. Gaussian smoothing, FWHM `gaussblr` mm
5. HU: `HU = µ · HU_calibration − 1000`, int16

## Coordinate conventions (established empirically vs the vendor nii)

- RTK's rotation axis is y, the NIfTI axial axis is z → the RTK volume is
  built as (x, z, y) and the array is reordered afterwards.
- The MILabs volume frame is the **mirror of the RTK world frame in all three
  axes**: the grid is laid out at the mirrored origin `-(o + (n-1)·s)` and the
  reconstructed array reversed along every axis. Verified against
  `CT_2026-07-17_14h38_milabs.nii`: r = 0.993 at 4× preview with zero voxel
  residual shift.
- The nii header origin (`niiorigin` in the log) is cosmetic and differs from
  the reconstruction-grid origin (`origin`); both are honored.

## CUDA

Reconstruction runs on the GPU (like the vendor's `hardware=cuda`) when the
CUDA build of RTK is installed: `pip install itk-rtk-cuda121` (replaces
`itk-rtk`; needs the CUDA 12.1 toolkit in `C:\Program Files\NVIDIA GPU
Computing Toolkit`, whose `bin` dir the module registers via
`os.add_dll_directory` — Python does not search PATH for DLLs). Falls back to
CPU FDK automatically if the CUDA filters are unavailable. Full scan
(360 × 972×768 → 400×400×457) runs in ~80 s end to end on a Quadro P4000,
most of it TIFF loading; the CPU path takes tens of minutes.

Numpy arrays are moved in/out of `itk.CudaImage` through the host buffer
pointer (`ctypes.memmove` + `GetBufferPointer()`), since the pip wheels wrap
no cast/graft/array path for CudaImage.

**Known RTK bug**: the CUDA FDK filter leaks ~0.8 GB of GPU memory per
reconstruction (internal FFT buffers; unaffected by filter reuse or any
explicit release call). A long batch in one process therefore dies with
`CUFFT ERROR #5` / `CUDA ERROR: out of memory` after ~9 phases on the 8 GB
P4000. The CLI works around it by running batches in child processes of
`--phases-per-process` (default 4) phases each — the leak is bounded per
child and the OS reclaims everything at child exit.

Verified: full-resolution CUDA output vs vendor nii — r = 0.996, identical
value range, body voxels median |diff| = 1 HU (RMSE 6.9 HU).

## Python environment / Jupyter kernel

Everything here runs on the system **Python 3.10**
(`C:\Users\milabs\AppData\Local\Programs\Python\Python310`) — that's where
itk 5.4.6 + itk-rtk-cuda121 live. The conda envs (`mi-env`, `recon`, …) have
their own incompatible itk and **crash the kernel** if used.

In Jupyter, select the kernel **"Python 3.10 (RTK recon)"** (kernelspec name
`rtkrecon`) for any notebook that imports `milabs_rtk_recon`. The tutorial
notebook is already stamped with it. From a shell, `py` launches the right
Python.

## Gotcha: import order

`itk`/`RTK` must be imported and used **before** `pandas`/`nibabel`, otherwise
RTK filter construction crashes with a Windows access violation (pip
`itk-rtk` 2.7.0, itk 5.4.6, Python 3.10). `milabs_rtk_recon` handles this by
importing itk first — import it before anything else in your own drivers.
