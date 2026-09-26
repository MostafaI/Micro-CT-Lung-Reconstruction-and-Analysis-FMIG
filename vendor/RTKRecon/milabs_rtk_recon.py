"""FDK cone-beam CT reconstruction of MILabs U-CT data with RTK.

Reconstructs the corrected projections in <phase_dir>/ct-data/corr using the
geometry in calibration/geopar.cfg (and, when present, the parameter dump that
the MILabs reconstructor writes to Results/CT_*.log) so that the output matches
the vendor reconstruction voxel-for-voxel.

Pipeline:
  corrected TIFFs -> line integrals ln(I0/I) -> water precorrection polynomial
  -> RTK FDK (Hann-windowed ramp) -> Gaussian smoothing -> HU -> int16 NIfTI

Also exports the per-projection 3x4 projection matrices (world -> detector mm,
and world -> detector pixel) and the RTK geometry XML.

Usage:
  py milabs_rtk_recon.py "D:\\Data\\PhNd6\\2026-07-17_14h38\\Optimized_Reconstruction\\Phase_0"
"""

import argparse
import re
from pathlib import Path

# The CUDA build of RTK loads cudart/cufft from the CUDA toolkit; Python 3.8+
# does not search PATH for DLL dependencies, so register the toolkit bin dir.
import os as _os
import glob as _glob
for _d in sorted(_glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin")):
    _os.add_dll_directory(_d)

# itk/RTK must be imported (and a first object constructed) BEFORE pandas or
# nibabel: those pull in a conflicting DLL that makes RTK filter construction
# crash with an access violation (pip itk-rtk 2.7 / itk 5.4.6 on Windows).
# Import this module first, or import itk before pandas/nibabel, in any driver.
import sys as _sys
import itk
from itk import RTK as rtk

HAVE_CUDA = None  # resolved by _warm_up()


def _warm_up():
    """First template instantiation makes ITK load its whole DLL chain: ~40 s,
    once per process. Done lazily so that orchestration (phase discovery,
    subprocess chunking) stays instant; say so, instead of looking hung."""
    global HAVE_CUDA
    if HAVE_CUDA is not None:
        return
    print("loading ITK/RTK libraries (one-time, ~40 s) ...", file=_sys.stderr, flush=True)
    f3 = itk.Image[itk.F, 3]
    rtk.ConstantImageSource[f3].New()
    rtk.FDKConeBeamReconstructionFilter[f3].New()
    HAVE_CUDA = hasattr(rtk, "CudaFDKConeBeamReconstructionFilter")
    print("libraries loaded.", file=_sys.stderr, flush=True)

import numpy as np
import pandas as pd
import tifffile
import nibabel as nib
from tqdm import tqdm


# ---------------------------------------------------------------- parsing ----

def parse_kv_file(path):
    """Parse key=value files (geopar.cfg, .mCT, MILabs CT log). Skips comments.

    Values: float, comma-separated tuple of floats, or string. Bare flags -> True.
    """
    params = {}
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if "=" not in line:
            params[line] = True
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"')
        if "," in v:
            try:
                params[k] = tuple(float(x) for x in v.split(","))
                continue
            except ValueError:
                pass
        try:
            params[k] = float(v)
        except ValueError:
            params[k] = v
    return params


def find_inputs(phase_dir):
    phase_dir = Path(phase_dir)
    ct_data = phase_dir / "ct-data"
    paths = {
        "phase_dir": phase_dir,
        "corr": ct_data / "corr",
        "geopar": ct_data / "calibration" / "geopar.cfg",
        "results": phase_dir / "Results",
    }
    logs = sorted(paths["results"].glob("CT_*.log"))
    if not logs:
        # The vendor writes its parameter log only for phases it reconstructed,
        # but the parameters (volume origin included) are scan-wide: borrow the
        # log from a sibling phase so every phase lands on the same voxel grid.
        logs = sorted(phase_dir.parent.glob("Phase_*/Results/CT_*.log"))
    paths["milabs_log"] = logs[0] if logs else None
    csvs = sorted(ct_data.glob("proj_*_log.csv"))
    paths["proj_log"] = csvs[0] if csvs else None
    mcts = sorted(phase_dir.glob("*.mCT"))
    paths["mct"] = mcts[0] if mcts else None
    return paths


def build_config(paths, overrides=None):
    """Merge geopar.cfg, .mCT and (if present) the MILabs recon log into one dict."""
    cfg = {
        # defaults used when no MILabs log is available
        "i0": 60000.0,
        "hann": 1.0,
        "gaussblr": 0.32,          # FWHM in mm of post-recon Gaussian
        "spacing": 0.16,
        "dimension": (400.0, 400.0, 457.0),
        "wpc": (0.0, 1.0, 0.0),
        "hu": 20274.55556,
        "first_angle": 0.0,
    }
    cfg.update(parse_kv_file(paths["geopar"]))
    if paths["mct"]:
        mct = parse_kv_file(paths["mct"])
        cfg.setdefault("hu", mct.get("HU_calibration", cfg["hu"]))
        if "BeamHardening" in mct:
            cfg["wpc"] = (0.0, 1.0, float(mct["BeamHardening"]))
    if paths["milabs_log"]:
        log = parse_kv_file(paths["milabs_log"])
        for k in ("sdd", "sid", "proj_iso_x", "proj_iso_y", "source_x", "source_y",
                  "in_angle", "first_angle", "i0", "hann", "gaussblr", "spacing",
                  "dimension", "origin", "niiorigin", "wpc", "hu"):
            if k in log:
                cfg[k] = log[k]
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    # derived defaults when the log did not provide them
    dim = [int(d) for d in cfg["dimension"]]
    s = float(cfg["spacing"])
    cfg["dimension"] = dim
    if "origin" not in cfg:
        cfg["origin"] = tuple(-(n - 1) * s / 2.0 for n in dim)
    if "niiorigin" not in cfg:
        cfg["niiorigin"] = cfg["origin"]
    return cfg


def get_angles(proj_log_csv, first_angle):
    """Gantry angles in degrees: first_angle + measured angle (3rd CSV column)."""
    df = pd.read_csv(proj_log_csv, header=None)
    return float(first_angle) + df.iloc[:, 2].to_numpy(dtype=float)


# ------------------------------------------------------------ projections ----

def load_line_integrals(corr_dir, i0, wpc, downsample=1):
    """Read corrected TIFFs, convert to line integrals with water precorrection.

    Returns an ITK float image stack (projection index along 3rd dim) whose
    origin is 0 at the first detector pixel; geopar proj_iso_* then position the
    detector via the RTK projection offsets.
    """
    files = sorted(Path(corr_dir).glob("*.tif")) + sorted(Path(corr_dir).glob("*.tiff"))
    if not files:
        raise FileNotFoundError(f"no TIFF projections in {corr_dir}")
    with tifffile.TiffFile(files[0]) as tf:
        xr = tf.pages[0].tags["XResolution"].value
        yr = tf.pages[0].tags["YResolution"].value
    su = 25.4 * xr[1] / xr[0]  # mm / pixel
    sv = 25.4 * yr[1] / yr[0]

    stack = np.stack([
        tifffile.imread(str(f))
        for f in tqdm(files, desc="loading projections", unit="img",
                      disable=None)
    ]).astype(np.float32)
    import time
    t0 = time.time()
    tqdm.write("computing line integrals (log + water precorrection) ...")
    # all in-place float32: polyval would promote the ~1 GB stack to float64
    p = np.clip(stack, 1.0, None, out=stack)
    np.divide(np.float32(i0), p, out=p)
    np.log(p, out=p)
    coeffs = list(wpc)
    res = np.full_like(p, coeffs[-1])          # Horner evaluation
    for c in reversed(coeffs[:-1]):
        res *= p
        res += np.float32(c)
    p = res
    tqdm.write(f"line integrals done ({time.time() - t0:.0f} s)")

    ou = ov = 0.0
    if downsample > 1:
        d = downsample
        n, h, w = p.shape
        p = p[:, : h // d * d, : w // d * d]
        p = p.reshape(n, h // d, d, w // d * d).mean(2)
        p = p.reshape(n, h // d, w // d, d).mean(3)
        # block centers sit (d-1)/2 native pixels inside; keep physical positions
        ou, ov = (d - 1) / 2.0 * su, (d - 1) / 2.0 * sv
        su, sv = su * d, sv * d

    img = itk.image_from_array(np.ascontiguousarray(p))
    img.SetSpacing([float(su), float(sv), 1.0])
    img.SetOrigin([float(ou), float(ov), 0.0])
    return img


# --------------------------------------------------------------- geometry ----

def build_geometry(cfg, angles):
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for a in angles:
        geometry.AddProjection(
            float(cfg["sid"]), float(cfg["sdd"]), float(a),
            float(cfg["proj_iso_x"]), float(cfg["proj_iso_y"]),
            0.0,                                  # out-of-plane angle
            float(cfg.get("in_angle", 0.0)),
            float(cfg.get("source_x", 0.0)), float(cfg.get("source_y", 0.0)),
        )
    return geometry


def export_geometry(geometry, proj_img, out_dir, stem="geometry"):
    """Write RTK geometry XML plus per-projection 3x4 projection matrices."""
    out_dir = Path(out_dir)
    writer = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
    writer.SetObject(geometry)
    writer.SetFilename(str(out_dir / f"{stem}.xml"))
    writer.WriteFile()

    n = len(geometry.GetGantryAngles())
    mats = np.stack([itk.array_from_matrix(geometry.GetMatrix(i)) for i in range(n)])

    # world (mm) -> detector pixel index: fold in projection origin/spacing
    ou, ov = proj_img.GetOrigin()[0], proj_img.GetOrigin()[1]
    su, sv = proj_img.GetSpacing()[0], proj_img.GetSpacing()[1]
    to_pix = np.array([[1 / su, 0, -ou / su],
                       [0, 1 / sv, -ov / sv],
                       [0, 0, 1.0]])
    mats_pix = np.einsum("ij,njk->nik", to_pix, mats)

    np.save(out_dir / f"{stem}_matrices_mm.npy", mats)
    np.save(out_dir / f"{stem}_matrices_pix.npy", mats_pix)
    with open(out_dir / f"{stem}_matrices.txt", "w") as f:
        f.write("# MILabs U-CT RTK projection matrices, one 3x4 block per projection.\n"
                "# P_mm maps homogeneous world coords (mm) to detector (u,v,w) in mm;\n"
                "# P_pix maps to detector pixel indices. Divide by 3rd row to dehomogenize.\n")
        for i in range(n):
            f.write(f"\n# projection {i}\nP_mm:\n{mats[i]}\nP_pix:\n{mats_pix[i]}\n")
    return mats, mats_pix


# ----------------------------------------------------------------- recon -----

def _cuda_image_from_itk(img):
    """Copy a CPU itk.Image[F,3] into an itk.CudaImage[F,3].

    No cast/graft path is wrapped in the pip wheels; copy through the host
    buffer pointer (GetBufferPointer marks the GPU buffer stale, so the data
    is uploaded on first GPU use).
    """
    import ctypes
    tqdm.write("staging projections for the GPU ...")
    arr = itk.array_view_from_image(img)
    cu = itk.CudaImage[itk.F, 3].New()
    cu.SetRegions(img.GetLargestPossibleRegion())
    cu.Allocate()
    cu.SetSpacing(img.GetSpacing())
    cu.SetOrigin(img.GetOrigin())
    cu.SetDirection(img.GetDirection())
    src = np.ascontiguousarray(arr, dtype=np.float32)
    ctypes.memmove(int(cu.GetBufferPointer()), src.ctypes.data, src.nbytes)
    return cu


def _array_from_cuda_image(cu, shape_zyx):
    import ctypes
    out = np.empty(shape_zyx, dtype=np.float32)
    ctypes.memmove(out.ctypes.data, int(cu.GetBufferPointer()), out.nbytes)
    return out


def fdk_reconstruct(geometry, proj_img, size_xyz, spacing, origin_xyz,
                    hann=1.0, hardware="cuda"):
    """Run RTK FDK. size/origin given in NIfTI (x,y,z) order; RTK's rotation
    axis is y, so the volume is built as (x, z, y) in RTK world coordinates and
    the returned numpy array is reordered to (x, y, z)."""
    nx, ny, nz = size_xyz
    ox, oy, oz = origin_xyz
    use_cuda = hardware == "cuda" and HAVE_CUDA
    ImageType = itk.CudaImage[itk.F, 3] if use_cuda else itk.Image[itk.F, 3]

    src = rtk.ConstantImageSource[ImageType].New()
    src.SetSize([int(nx), int(nz), int(ny)])          # RTK: (x, axial=y, z)
    src.SetSpacing([float(spacing)] * 3)
    src.SetOrigin([float(ox), float(oz), float(oy)])
    src.SetConstant(0.0)

    if use_cuda:
        fdk = rtk.CudaFDKConeBeamReconstructionFilter.New()
        fdk.SetInput(1, _cuda_image_from_itk(proj_img))
    else:
        fdk = rtk.FDKConeBeamReconstructionFilter[ImageType].New()
        fdk.SetInput(1, proj_img)
    fdk.SetInput(0, src.GetOutput())
    fdk.SetGeometry(geometry)
    fdk.GetRampFilter().SetHannCutFrequency(float(hann))

    bar = tqdm(total=100, desc=f"FDK ({'cuda' if use_cuda else 'cpu'})",
               unit="%", disable=None)
    cmd = itk.PyCommand.New()
    cmd.SetCommandCallable(lambda: (setattr(bar, "n", int(fdk.GetProgress() * 100)),
                                    bar.refresh()))
    fdk.AddObserver(itk.ProgressEvent(), cmd)
    fdk.Update()
    bar.n = 100
    bar.close()
    # Break the fdk -> observer -> lambda -> fdk reference cycle: it keeps the
    # filter (and its GPU buffers) alive across phases until a GC cycle pass,
    # which exhausts VRAM in multi-phase batches.
    fdk.RemoveAllObservers()

    if use_cuda:                                      # (z_rtk, y_rtk, x_rtk)
        arr = _array_from_cuda_image(fdk.GetOutput(), (int(ny), int(nz), int(nx)))
    else:
        arr = itk.array_from_image(fdk.GetOutput())
    # -> (x, y_nii=z_rtk, z_nii=y_rtk)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def smooth_and_scale(mu_xyz, spacing, gauss_fwhm_mm, hu_scale, out_dtype=np.int16):
    """Gaussian smoothing (FWHM in mm) then mu -> HU (air=-1000)."""
    img = itk.image_from_array(np.ascontiguousarray(mu_xyz.transpose(2, 1, 0)))
    img.SetSpacing([float(spacing)] * 3)
    if gauss_fwhm_mm and gauss_fwhm_mm > 0:
        sigma_mm = gauss_fwhm_mm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        sm = itk.SmoothingRecursiveGaussianImageFilter.New(img)
        sm.SetSigma(float(sigma_mm))
        sm.Update()
        img = sm.GetOutput()
    mu = itk.array_from_image(img).transpose(2, 1, 0)
    hu = mu * float(hu_scale) - 1000.0
    info = np.iinfo(out_dtype)
    return np.clip(np.rint(hu), info.min, info.max).astype(out_dtype)


def write_nifti(data_xyz, spacing, nii_origin, out_path):
    affine = np.diag([spacing, spacing, spacing, 1.0])
    affine[:3, 3] = nii_origin
    nib.save(nib.Nifti1Image(np.asfortranarray(data_xyz), affine), str(out_path))


# ------------------------------------------------------------------- main ----

def reconstruct_phase(phase_dir, out_name=None, downsample=1, vol_scale=1,
                      overrides=None, save=True, export_matrices=True):
    """Full pipeline for one Phase_* directory. Returns (hu_volume_xyz, cfg).

    downsample / vol_scale > 1 give a fast preview reconstruction.
    """
    _warm_up()
    paths = find_inputs(phase_dir)
    cfg = build_config(paths, overrides)
    angles = get_angles(paths["proj_log"], cfg["first_angle"])

    proj = load_line_integrals(paths["corr"], cfg["i0"], cfg["wpc"], downsample)
    geometry = build_geometry(cfg, angles)

    dim = [max(1, n // vol_scale) for n in cfg["dimension"]]
    spacing = float(cfg["spacing"]) * vol_scale
    # keep the voxel-center grid aligned with the vendor grid when coarsening
    origin = [o + (vol_scale - 1) / 2.0 * float(cfg["spacing"]) for o in cfg["origin"]]
    # The MILabs volume frame is the mirror of the RTK world frame in all three
    # axes (established against the vendor reconstruction): lay the grid out
    # mirrored, reconstruct, then reverse the array along every axis.
    rtk_origin = [-(o + (n - 1) * spacing) for o, n in zip(origin, dim)]

    mu = fdk_reconstruct(geometry, proj, dim, spacing, rtk_origin,
                         hann=cfg["hann"], hardware=cfg.get("hardware", "cuda"))
    mu = mu[::-1, ::-1, ::-1]
    tqdm.write("smoothing + HU conversion ...")
    hu = smooth_and_scale(mu, spacing, float(cfg["gaussblr"]) / max(1, vol_scale),
                          cfg["hu"])

    if save:
        results = paths["results"]
        results.mkdir(exist_ok=True)
        scan = re.sub(r"^CT_|\.log$", "", paths["milabs_log"].name) \
            if paths["milabs_log"] else Path(phase_dir).name
        name = out_name or f"CT_{scan}_rtk.nii"
        nii_origin = [o + (vol_scale - 1) / 2.0 * float(cfg["spacing"])
                      for o in cfg["niiorigin"]]
        write_nifti(hu, spacing, nii_origin, results / name)
        if export_matrices:
            export_geometry(geometry, proj, results, stem=f"CT_{scan}_rtk_geometry")
        print(f"wrote {results / name}")
    return hu, cfg


def find_phase_dirs(path):
    """A phase dir itself -> [it]; a dir containing Phase_* -> all of them,
    numerically sorted."""
    path = Path(path)
    if (path / "ct-data" / "corr").is_dir():
        return [path]

    def num(d):
        m = re.search(r"(\d+)$", d.name)
        return int(m.group(1)) if m else -1

    return sorted((d for d in path.glob("Phase_*")
                   if (d / "ct-data" / "corr").is_dir()), key=num)


def main():
    import gc
    import subprocess
    import sys
    import time
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+",
                    help=r"Phase_N dir(s), or an Optimized_Reconstruction dir "
                         r"(reconstructs every phase in it)")
    ap.add_argument("-o", "--out-name", default=None, help="output NIfTI filename")
    ap.add_argument("--downsample", type=int, default=1,
                    help="detector downsampling factor (preview)")
    ap.add_argument("--vol-scale", type=int, default=1,
                    help="volume coarsening factor (preview)")
    ap.add_argument("--hann", type=float, default=None)
    ap.add_argument("--gaussblr", type=float, default=None)
    ap.add_argument("--no-matrices", action="store_true")
    ap.add_argument("--phases-per-process", type=int, default=4,
                    help="RTK's CUDA FDK leaks ~0.8 GB GPU memory per "
                         "reconstruction; phases run in child processes of "
                         "this many phases so the leak never exhausts VRAM")
    # internal (set when spawning children): global numbering for progress
    ap.add_argument("--index-offset", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--index-total", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    phases = []
    for p in args.paths:
        phases += find_phase_dirs(p)
    if not phases:
        sys.exit(f"error: no ct-data/corr under {args.paths} or their Phase_* subdirs")

    t0 = time.time()

    # Too many phases for one process (GPU leak, see --phases-per-process):
    # orchestrate child processes, each handling a chunk.
    if len(phases) > args.phases_per_process:
        n = args.phases_per_process
        print(f"{len(phases)} phases: running in child processes of {n} "
              f"(GPU memory hygiene)", flush=True)
        failed = 0
        for start in range(0, len(phases), n):
            chunk = phases[start:start + n]
            cmd = [sys.executable, "-u", __file__, *(str(d) for d in chunk),
                   "--phases-per-process", str(n),
                   "--index-offset", str(start), "--index-total", str(len(phases))]
            if args.out_name:
                cmd += ["-o", args.out_name]
            if args.downsample != 1:
                cmd += ["--downsample", str(args.downsample)]
            if args.vol_scale != 1:
                cmd += ["--vol-scale", str(args.vol_scale)]
            if args.hann is not None:
                cmd += ["--hann", str(args.hann)]
            if args.gaussblr is not None:
                cmd += ["--gaussblr", str(args.gaussblr)]
            if args.no_matrices:
                cmd += ["--no-matrices"]
            failed += subprocess.run(cmd).returncode
        print(f"\ndone: {len(phases) - failed}/{len(phases)} phases "
              f"in {int(time.time() - t0) // 60} min", flush=True)
        sys.exit(1 if failed else 0)

    total = args.index_total or len(phases)
    failed = []
    for i, d in enumerate(phases, 1 + args.index_offset):
        if total > 1:
            print(f"\n=== [{i}/{total}] {d.name} ===", flush=True)
        try:
            reconstruct_phase(d, out_name=args.out_name,
                              downsample=args.downsample, vol_scale=args.vol_scale,
                              overrides={"hann": args.hann, "gaussblr": args.gaussblr},
                              export_matrices=not args.no_matrices)
        except Exception as e:
            print(f"*** {d.name} FAILED: {e}", file=sys.stderr, flush=True)
            failed.append(d.name)
        # free host + GPU memory of this phase before starting the next
        gc.collect()

    if failed:
        print(f"failed: {' '.join(failed)}", file=sys.stderr, flush=True)
    sys.exit(len(failed))


if __name__ == "__main__":
    main()
