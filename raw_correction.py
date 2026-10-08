"""
raw_correction.py

Replicates MILabs' U-CT projection correction step: turns the raw
ct-data/Chunk*.tif acquisitions into dark/white-field corrected
ct-data/corr/proj_000_0_NNNNNNNN.tif projections, using the reference
frames in ct-data/calibration. This lets the pipeline start from raw scanner
data instead of needing the vendor software to produce ct-data/corr first.

Ported from Desktop/Pipeline/Correction/correction.py (same algorithm), with
two app-specific changes: a thread pool instead of a process pool (worker
processes spawned from pipeline_driver.py would each re-import the whole
driver, torch included), and output written to a temporary folder that is
only renamed to ct-data/corr once every projection has been written.

Each Chunk*.tif is a multi-page stack of sub-frames taken at one rotation
angle. Every sub-frame is corrected individually and numbered sequentially
across the whole scan: chunk c's sub-frame j becomes global projection
index c * frames_per_chunk + j (chunk 0 -> proj 0..31, chunk 1 -> 32..63,
and so on, for 32-frame chunks).

Exact algorithm (reverse-engineered against genuine MILabs corr output):

    gain  = float32(30000) / (float32(white) - float32(dark))   # per pixel
    half  = floor( (float32(raw) - float32(dark)) * gain )      # float32 mult
    value = 2 * clip(half, 0, 32767)          # even numbers only, max 65534
    image = rot90(value, k=-1)                # sensor portrait -> landscape
    image = roll(image, -1, axis=0)           # one-line readout shift

Every pixel matches MILabs' output bit-for-bit EXCEPT two cases the Chunk
files do not contain enough information to reproduce (~0.13% of pixels):

  1. Output row 767 (the last row): its true content is detector line 0
     sampled at a stream position the Chunk files do not record. The wrap
     fills it with the same line from the current sub-frame (statistically
     identical).
  2. Defect-map pixels (calibration/*Defect_Map_2x2.smv): MILabs interpolates
     these at full detector resolution BEFORE 2x2 binning. They are filled
     with the median of their valid 8 corrected neighbors.

Validated on D:\\Data\\PhNd5\\2026-07-15_11h24 (corr_old) and
D:\\Data\\R53\\2026-10-06_12h54 (corr_to_match): 11520 projections each,
99.8664% of pixels bit-exact, ZERO mismatches outside the two cases above.

Usage:
    python raw_correction.py "D:\\Data\\R53\\2026-10-06_12h54\\ct-data" [--validate-against <corr dir>]
"""
import os
import re
import glob
import time
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tifffile

CHUNK_RE = re.compile(r"Chunk(\d+)_(\d+)_(\d+)_(\d+)\.tif$", re.IGNORECASE)

HALF_SCALE = np.float32(30000.0)   # outputs are 2 * floor(HALF_SCALE * transmission)


def load_calibration(calibration_dir):
    dark_path = glob.glob(os.path.join(calibration_dir, "*_dark.tif"))
    white_path = glob.glob(os.path.join(calibration_dir, "*_white.tif"))
    defect_path = glob.glob(os.path.join(calibration_dir, "*Defect_Map*.smv"))
    if not dark_path or not white_path:
        raise FileNotFoundError(f"dark/white calibration tif not found in {calibration_dir}")

    dark = tifffile.imread(dark_path[0]).astype(np.float32)
    white = tifffile.imread(white_path[0]).astype(np.float32)
    gain = (HALF_SCALE / (white - dark)).astype(np.float32)
    defect_mask = load_defect_map(defect_path[0], dark.shape) if defect_path else None
    return dark, gain, defect_mask


def load_defect_map(smv_path, expected_shape):
    """MILabs .smv: 512-byte ASCII header, then a little-endian uint16 map
    the same size as a detector frame; 1 = defective pixel."""
    with open(smv_path, "rb") as f:
        header = f.read(512)
        data = np.frombuffer(f.read(), dtype="<u2")
    m = re.search(rb"SIZE1=(\d+)", header)
    n = re.search(rb"SIZE2=(\d+)", header)
    width, height = int(m.group(1)), int(n.group(1))
    mask = data.reshape(height, width) != 0
    if mask.shape != expected_shape:
        raise ValueError(f"defect map shape {mask.shape} != calibration frame shape {expected_shape}")
    return mask


def fill_defects(corrected, defect_mask):
    """Fill defective pixels with the median of their valid 8-neighbors in
    the corrected image (approximation - see module docstring)."""
    out = corrected.copy()
    h, w = corrected.shape
    for y, x in zip(*np.where(defect_mask)):
        y0, y1 = max(0, y - 1), min(h, y + 2)
        x0, x1 = max(0, x - 1), min(w, x + 2)
        nb = corrected[y0:y1, x0:x1]
        valid = nb[~defect_mask[y0:y1, x0:x1]]
        if valid.size:
            out[y, x] = np.median(valid)
    return out


def correct_subframe(raw_frame, dark, gain, defect_mask=None):
    """Bit-exact MILabs correction of one raw sensor sub-frame -> one
    landscape proj image (uint16, even values)."""
    v = ((raw_frame.astype(np.float32) - dark) * gain).astype(np.float32)
    # saturation is applied to the half-scale value (max 32767), so the
    # brightest possible output is 65534, not 65535
    q = 2.0 * np.clip(np.floor(v.astype(np.float64)), 0, 32767)
    if defect_mask is not None:
        q = fill_defects(q, defect_mask)
    out = np.roll(np.rot90(q, k=-1), -1, axis=0)
    return out.astype(np.uint16)


def defect_neighbor_index(defect_mask):
    """Precomputes, for every defective pixel, the flat indices of its valid
    (in-image, non-defective) 8-neighbors, padded with -1 to a fixed width
    of 8. Lets fill_defects_stack do the whole stack in a few numpy calls
    instead of a Python loop per pixel per frame."""
    h, w = defect_mask.shape
    ys, xs = np.where(defect_mask)
    nbr = np.full((len(ys), 8), -1, dtype=np.int64)
    for k, (y, x) in enumerate(zip(ys, xs)):
        idx = [yy * w + xx
               for yy in range(max(0, y - 1), min(h, y + 2))
               for xx in range(max(0, x - 1), min(w, x + 2))
               if not defect_mask[yy, xx]]
        nbr[k, :len(idx)] = idx
    return ys * w + xs, nbr


def fill_defects_stack(stack, defect_index):
    """Vectorized fill_defects over a (frames, H, W) stack: each defective
    pixel gets the median of its valid 8-neighbors (same values as
    fill_defects). Modifies stack in place."""
    targets, nbr = defect_index
    if targets.size == 0:
        return stack
    n = stack.shape[0]
    flat = stack.reshape(n, -1)
    vals = flat[:, np.maximum(nbr, 0)].astype(np.float64)   # (frames, n_defects, 8)
    vals[:, nbr < 0] = np.nan
    has_valid = (nbr >= 0).any(axis=1)
    med = np.nanmedian(vals[:, has_valid], axis=2)
    flat[:, targets[has_valid]] = med
    return stack


def correct_stack(stack, dark, gain, defect_index=None):
    """correct_subframe applied to a whole (frames, H, W) sub-frame stack at
    once, with identical results. float32 throughout is exact here: floor of
    a float32 is representable in float32, and 2 * [0, 32767] is exact."""
    v = (stack.astype(np.float32) - dark) * gain
    np.floor(v, out=v)
    np.clip(v, 0, 32767, out=v)
    v *= 2
    if defect_index is not None:
        fill_defects_stack(v, defect_index)
    out = np.roll(np.rot90(v, k=-1, axes=(1, 2)), -1, axis=1)
    return out.astype(np.uint16)


def fix_dead_pixels(im, pixels=1):
    """In-place replica of recon_functions_all.correct_corr_dir's per-image
    fix (the pipeline's former second pass over ct-data/corr): every 0-valued
    pixel, in np.where order, becomes the mean of the non-zero pixels in its
    (2*pixels+1)^2 neighborhood, truncated to uint16. Pixels fixed earlier in
    the scan count as non-zero neighbors for later ones, exactly as there.
    Returns True if anything was changed."""
    xs, ys = np.where(im == 0)
    if xs.size == 0:
        return False
    with np.errstate(invalid="ignore", divide="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-zero neighborhood -> nan -> 0, as before
            for x, y in zip(xs, ys):
                xmin = x - pixels if x > 0 else 0
                ymin = y - pixels if y > 0 else 0
                temp = im[xmin:x + pixels + 1, ymin:y + pixels + 1]
                im[x, y] = temp[temp != im[x, y]].mean()
    return True


def load_chunk(chunk_path):
    """Read a Chunk*.tif as its full (frames_per_chunk, H, W) sub-frame stack."""
    stack = tifffile.imread(chunk_path)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    return stack


def parse_chunk_filename(chunk_path):
    match = CHUNK_RE.search(os.path.basename(chunk_path))
    if not match:
        raise ValueError(f"'{chunk_path}' does not look like a Chunk*.tif file")
    series, translation, subseries, index = match.groups()
    return series, translation, subseries, int(index)


# MILabs tags every corr projection with its pixel pitch as a TIFF resolution
# in inches: detector pixel 0.0748 mm times the binning factor, stored as the
# rational 4294967295 / round(4294967295 * pitch_inch). Pipeline code relies on
# it - e.g. recon_functions_all saves imputed projections with
# dpi=template.info['dpi'], which raises KeyError without these tags.
DETECTOR_PIXEL_MM = 0.0748
DETECTOR_FULL_WIDTH = 1944   # unbinned landscape projection width


# Exact denominators found in MILabs corr files (1x1: 1944x1536 projections,
# 2x2: 972x768); the formula below is within 1-2 of these and is only used
# for any other binning.
VENDOR_RESOLUTION_DEN = {1: 12648170, 2: 25296341}


def resolution_tag(proj_width):
    binning = max(1, round(DETECTOR_FULL_WIDTH / proj_width))
    num = 4294967295
    den = VENDOR_RESOLUTION_DEN.get(binning)
    if den is None:
        den = int(round(num * DETECTOR_PIXEL_MM * binning / 25.4))
    return (num, den)


def write_projection(path, proj):
    """Writes one corr projection with the same tags MILabs uses (uint16,
    uncompressed, resolution in inches - see resolution_tag)."""
    r = resolution_tag(proj.shape[1])
    tifffile.imwrite(path, proj, resolution=(r, r), resolutionunit="INCH",
                     metadata=None, software=False)


def output_filename(series, translation, index):
    return f"proj_{series}_{translation}_{index:08d}.tif"


def find_chunks(ct_data_dir):
    """Chunk*.tif files in ct_data_dir, in acquisition order."""
    return sorted(glob.glob(os.path.join(ct_data_dir, "Chunk*.tif")),
                  key=lambda p: parse_chunk_filename(p)[3])


def has_raw_data(ct_data_dir):
    return bool(find_chunks(ct_data_dir)) and os.path.isdir(os.path.join(ct_data_dir, "calibration"))


def correct_ct_data(ct_data_dir, output_dir=None, max_workers=None, frames_per_chunk=None,
                    overwrite=False, progress_every=10, fix_dead=False):
    """Correct every Chunk*.tif in ct_data_dir into per-sub-frame proj_*.tif
    files in output_dir (default <ct_data_dir>/corr), numbered sequentially
    as chunk_position * frames_per_chunk + subframe_index.

    Projections are written to "<output_dir>_tmp" first and only renamed to
    output_dir once all of them are written, so a crash or cancel never
    leaves a half-filled corr folder that later steps would trust. If
    output_dir already exists and is non-empty it is left alone (returns
    without doing anything) unless overwrite=True.

    fix_dead=True also applies the pipeline's dead-pixel pass (see
    fix_dead_pixels) in the same pass, so the output equals what
    MILabs' corr would become after recon_functions_all.correct_corr_dir.

    frames_per_chunk defaults to the sub-frame count of the first chunk
    file and is assumed constant across the scan (32 for this scanner).
    Returns the output folder."""
    output_dir = os.path.normpath(output_dir or os.path.join(ct_data_dir, "corr"))
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        if not overwrite:
            print(f"{output_dir} already exists - skipping raw projection correction "
                  f"(delete or rename it to regenerate).", flush=True)
            return output_dir
        shutil.rmtree(output_dir)

    calibration_dir = os.path.join(ct_data_dir, "calibration")
    chunk_paths = find_chunks(ct_data_dir)
    if not chunk_paths:
        raise FileNotFoundError(f"no Chunk*.tif files found in {ct_data_dir}")
    if frames_per_chunk is None:
        frames_per_chunk = load_chunk(chunk_paths[0]).shape[0]
    dark, gain, defect_mask = load_calibration(calibration_dir)
    defect_index = defect_neighbor_index(defect_mask) if defect_mask is not None else None

    tmp_dir = output_dir + "_tmp"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)  # leftovers from an interrupted run
    os.makedirs(tmp_dir)

    def process_one(task):
        chunk_position, chunk_path = task
        series, translation, _subseries, _index = parse_chunk_filename(chunk_path)
        stack = load_chunk(chunk_path)
        if stack.shape[0] != frames_per_chunk:
            raise ValueError(f"{os.path.basename(chunk_path)} has {stack.shape[0]} sub-frames, "
                             f"expected {frames_per_chunk}")
        projs = correct_stack(stack, dark, gain, defect_index)
        for j in range(projs.shape[0]):
            if fix_dead:
                fix_dead_pixels(projs[j])
            name = output_filename(series, translation, chunk_position * frames_per_chunk + j)
            write_projection(os.path.join(tmp_dir, name), projs[j])

    n = len(chunk_paths)
    max_workers = max_workers or min(16, os.cpu_count() or 4)
    print(f"Correcting {n} raw chunks x {frames_per_chunk} sub-frames = {n * frames_per_chunk} "
          f"projections ({max_workers} threads) -> {output_dir}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for done, _ in enumerate(executor.map(process_one, enumerate(chunk_paths)), start=1):
            if done % progress_every == 0 or done == n:
                print(f"  corrected {done}/{n} chunks ({time.time() - t0:.0f} s)", flush=True)

    os.rename(tmp_dir, output_dir)
    print(f"Raw projection correction done in {(time.time() - t0) / 60:.1f} min.", flush=True)
    return output_dir


def compare_to_reference(output_dir, reference_dir, defect_mask=None):
    """Report how closely output_dir matches proj_*.tif files of the same
    name in reference_dir, splitting out the two documented approximation
    zones (last output row and defect pixels). Returns the number of
    mismatches outside those zones plus missing files (0 = correct)."""
    ref_paths = sorted(glob.glob(os.path.join(reference_dir, "proj_*.tif")))
    if not ref_paths:
        raise FileNotFoundError(f"no proj_*.tif files found in {reference_dir}")

    defect_out = None
    if defect_mask is not None:
        defect_out = np.roll(np.rot90(defect_mask, k=-1), -1, axis=0)

    n_total = n_exact = n_last_row_bad = n_defect_bad = n_other_bad = n_files = n_missing = 0
    for ref_path in ref_paths:
        out_path = os.path.join(output_dir, os.path.basename(ref_path))
        if not os.path.isfile(out_path):
            n_missing += 1
            continue
        ref = tifffile.imread(ref_path).astype(np.int64)
        out = tifffile.imread(out_path).astype(np.int64)
        bad = ref != out
        n_files += 1
        n_total += bad.size
        n_exact += int(bad.size - bad.sum())
        last_row = np.zeros_like(bad)
        last_row[-1, :] = True
        n_last_row_bad += int((bad & last_row).sum())
        if defect_out is not None:
            n_defect_bad += int((bad & defect_out & ~last_row).sum())
            n_other_bad += int((bad & ~last_row & ~defect_out).sum())
        else:
            n_other_bad += int((bad & ~last_row).sum())

    print(f"Compared {n_files} projections against {reference_dir} ({n_missing} missing from output)")
    if n_files:
        print(f"  bit-exact pixels: {n_exact}/{n_total} ({100.0 * n_exact / n_total:.4f}%)")
        print(f"  mismatches in last row (missing stream line, approximated): {n_last_row_bad}")
        print(f"  mismatches at defect-map pixels (approximated):            {n_defect_bad}")
        print(f"  mismatches elsewhere (should be 0):                        {n_other_bad}")
    return n_other_bad + n_missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ct_data_dir", help="ct-data folder containing Chunk*.tif and calibration/")
    parser.add_argument("--output", help="Output folder (default: <ct_data_dir>/corr)")
    parser.add_argument("--workers", type=int, default=None, help="Worker threads (default: min(16, CPUs))")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing non-empty output folder")
    parser.add_argument("--fix-dead", action="store_true",
                        help="Also fill 0-valued dead pixels (the pipeline's correct_corr_dir pass)")
    parser.add_argument("--validate-against", help="Existing MILabs corr folder to compare output against")
    args = parser.parse_args()

    output_dir = correct_ct_data(args.ct_data_dir, output_dir=args.output,
                                 max_workers=args.workers, overwrite=args.overwrite,
                                 fix_dead=args.fix_dead)
    if args.validate_against:
        _, _, defect_mask = load_calibration(os.path.join(args.ct_data_dir, "calibration"))
        return 1 if compare_to_reference(output_dir, args.validate_against, defect_mask) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
