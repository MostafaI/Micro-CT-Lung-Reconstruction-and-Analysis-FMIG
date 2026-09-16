"""
pipeline_driver.py

Headless, script form of Desktop/Pipeline/Automated-FMIG-Rat.ipynb.
Runs the exact same steps, in the exact same order, using the same functions
from the same modules the notebook uses - just without Jupyter, and with
plain-text progress markers on stdout so a GUI can follow along.

Usage:
    python pipeline_driver.py "D:\\Data\\Txk5\\2026-08-06_10h47" [--steps recon,segment,analysis,register]

    Group mode - runs the same steps across every session found under one or
    more rat directories (see get_rat_session_dirs / run_group), one rat at a
    time, one session at a time within each rat - same idea as
    Desktop/Pipeline/Rat_all_dates_analysis.ipynb's per-rat loops. Repeat
    --rats for multiple rats; mutually exclusive with the positional main_dir:
        python pipeline_driver.py --rats "D:\\Data\\PhNd7" --rats "D:\\Data\\PhNd9" --steps diaphragm

Must be run with the "mi-env" conda environment's python.exe - the same
kernel the notebook itself uses (see kernel.json / recon_phase.bat).
"""
import argparse
import os
import sys
import time
import traceback

# Headless plotting: the notebook cells below call plt.show()/plt.savefig().
# This MUST happen before anything else imports matplotlib.pyplot, otherwise
# a real GUI backend gets selected first and plt.show() will hang forever
# waiting for a window that nothing will ever close.
import matplotlib
matplotlib.use("Agg")

# The notebook and its sibling modules (recon_functions_all.py, Analysis.py,
# utils.py, ...) live one folder up from this script - the repo root - and
# are imported via plain "from X import *". Put that folder on sys.path first.
PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)
os.chdir(PIPELINE_DIR)  # some helper functions use relative/CWD-relative paths

STEP_ORDER = ["recon", "segment", "analysis", "register", "diaphragm"]


def log_step(name):
    print(f"\n===STEP=== {name}", flush=True)


def log_artifact(path):
    if path and os.path.exists(path):
        print(f"===ARTIFACT=== {os.path.abspath(path)}", flush=True)


def log_done(name):
    print(f"===STEP_DONE=== {name}", flush=True)


def log_failed(name, exc):
    print(f"===STEP_FAILED=== {name}: {exc}", flush=True)
    traceback.print_exc()


# --------------------------------------------------------------------------
# Everything below this line is copied as-is from the "Imports and Functions"
# cell of Automated-FMIG-Rat.ipynb, so behavior matches the notebook exactly.
# --------------------------------------------------------------------------
from recon_functions_all import *
from recon_functions import *
from Analysis import *
from utils import *
from segment_from_projections import find_lungs
import subprocess as _subprocess  # keep builtin 'subprocess' import below too

import subprocess
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def step_2(main_dir):
    folder1 = os.path.join(main_dir, 'ct-data', 'corr')
    corr_dir = os.path.join(main_dir, 'ct-data', 'corr')
    folder2 = os.path.join(main_dir, 'ct-data', 'corr_new')
    if not os.path.isdir(folder2):
        os.mkdir(folder2)
    t = time.time()
    if not os.path.isdir(folder1 + '_old'):
        correct_corr_dir(folder1, out_dir=folder2)
        if os.path.exists(folder2):
            os.rename(folder1, folder1 + '_old')
            os.rename(folder2, folder1)
    print('Took', round((time.time() - t) / 60, 2), 'minutes')
    print('Step 2 is complete')


def step_3(main_dir, corr_dir, marign=75, threshold=0.5):
    sagittal_slice_num, coronal_slice_num = 5000, 2502
    tiff_path = corr_dir + '/proj_000_0_000' + "{:05d}".format(sagittal_slice_num) + '.tif'
    mask_pred, diaphragm_x, diaphragm_y_smooth, diaphragm_height = find_lungs(tiff_path)

    coronal_tiff_path = corr_dir + '/proj_000_0_000' + "{:05d}".format(coronal_slice_num) + '.tif'
    mask_pred_coronal, _, _, _ = find_lungs(coronal_tiff_path)

    try:
        diaphragm_height = int(diaphragm_height)
        xlimits = [diaphragm_height - marign, diaphragm_height + marign]  # Default is 350 to 550
    except Exception as e:
        print(f'Error: {e}')
        print('Fall back to using best model (attempt 2)')
        mask_pred, diaphragm_x, diaphragm_y_smooth, diaphragm_height = find_lungs(
            tiff_path,
            CHECKPOINT=r"C:\Users\milabs\Desktop\Pipeline\Rat-lung-segmentation-from-xray\checkpoints\best_model.pt")
        diaphragm_height = int(diaphragm_height)
        xlimits = [diaphragm_height - marign, diaphragm_height + marign]

        mask_pred_coronal, _, _, _ = find_lungs(
            coronal_tiff_path,
            CHECKPOINT=r"C:\Users\milabs\Desktop\Pipeline\Rat-lung-segmentation-from-xray\checkpoints\best_model.pt")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    im = get_image(corr_dir, slice_num=coronal_slice_num, normalize=True, threshold=threshold)

    plt.imshow(im, cmap='gray')
    plt.imshow(mask_pred_coronal, cmap="jet", alpha=0.25 * mask_pred, vmin=0, vmax=1)
    plt.axhline(y=diaphragm_height, color="green", linewidth=2.5)
    plt.axhline(y=xlimits[0], c='r', linestyle='--')
    plt.axhline(y=xlimits[1], c='r', linestyle='--')

    plt.subplot(1, 2, 2)
    im = get_image(corr_dir, slice_num=sagittal_slice_num, normalize=True, threshold=threshold)
    plt.imshow(im, cmap='gray')
    plt.imshow(mask_pred, cmap="jet", alpha=0.25 * mask_pred, vmin=0, vmax=1)
    plt.plot(diaphragm_x, diaphragm_y_smooth, color="yellow", linewidth=2.5, label="diaphragm (lung inferior border)")
    plt.axhline(y=diaphragm_height, color="green", linewidth=2.5)
    plt.axhline(y=xlimits[0], c='r', linestyle='--')
    plt.axhline(y=xlimits[1], c='r', linestyle='--')
    out_png = os.path.join(main_dir, 'MI_Rat_window_result.png')
    plt.savefig(out_png, dpi=100, bbox_inches='tight')
    plt.close()
    log_artifact(out_png)
    return xlimits


def step_4(main_dir, xlimits, threshold):
    ts = time.time()
    s, t, a = get_signal_from_image(os.path.join(main_dir, 'ct-data', 'corr'), subtract_baseline=1,
                                     spring=False, rabbit=False, xlimits=xlimits, threshold=threshold)
    print(f'Took {round((time.time() - ts) / 60, 2)} minutes.')
    plt.figure(figsize=(7, 4))
    plt.plot(t, s)
    plt.close()
    plt.figure(figsize=(7, 4))
    b = fancy_binning(a, s, t, flip_signal=False, only_phase_binning=0)
    b.nbins = 16
    b.max_freq = 100
    b.binning(plot=1, withline=True)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.2])
    out_png = os.path.join(main_dir, 'clustering_result.png')
    plt.savefig(out_png, dpi=100, bbox_inches='tight')
    plt.close()
    log_artifact(out_png)
    b.get_approximate_breathing_rate()
    return s, t, a

def clean_study_folder(session):
    recon_dirs = [os.path.join(session,x) for x in os.listdir(session) if 'Optimized_Reconstruction' in x]
    old_corr_dir = os.path.join(session, 'ct-data', 'corr_old')
    print('Cleaning: ', end = '')
    if os.path.isdir(old_corr_dir): 
        shutil.rmtree(old_corr_dir)
        print('*', end='')
    for recon_dir in recon_dirs:
        phase_dirs = [os.path.join(recon_dir, x) for x in os.listdir(recon_dir) if 'Phase_' in x]
        for phase_dir in phase_dirs: 
            shutil.rmtree(phase_dir)
            print('*', end='')
            pt = True
    print()
            
def MI_reconstruction(main_dir, threshold=0.5):
    corr_dir = os.path.join(main_dir, 'ct-data', 'corr')
    check_timing(main_dir)
    step_2(main_dir)
    xlimits = step_3(main_dir, corr_dir, threshold=threshold)
    s, t, a = step_4(main_dir, xlimits, threshold)
    create_milab_structure(main_dir,
                            intensity_phases=10,
                            time_phases=16,
                            spring=False,
                            s=s, t=t, a=a,
                            only_phase_binning=0,
                            fancy=1,
                            optim_SNR=False,
                            bm3d=False,
                            outname=outname,
                            phase=-1,
                            identifier='',
                            impute=True)
    root_dir = os.path.join(main_dir, outname)
    recon(root_dir)
    collect_results(main_dir, bm3d=False, mname=outname)
    make_gif(main_dir, outname, aslice=260, origin='lower')
    clean_study_folder(main_dir)
    log_artifact(os.path.join(main_dir, outname, 'Results', 'images', 'video.gif'))
    return


# RTKRecon is a separate, non-GitHub component that always lives under
# Desktop/Pipeline directly, regardless of where this script/repo sits.
RTKRECON_DIR = r"C:\Users\milabs\Desktop\Pipeline\RTKRecon"


def _recon_via_warm_server(ROOT_DIR):
    """Try the warm recon_server.py (see RTKRecon/recon_server.py) - skips the
    ~40s/process itk-rtk warm-up when it's already running. Raises
    ConnectionRefusedError if the server isn't up; caller falls back to the
    cold recon_phase.bat subprocess path."""
    if RTKRECON_DIR not in sys.path:
        sys.path.insert(0, RTKRECON_DIR)
    from recon_client import run_recon_via_server

    def on_line(text):
        print(text, end="")
        sys.stdout.flush()

    print("recon server detected - reconstructing on the warm process ...", flush=True)
    return run_recon_via_server(ROOT_DIR, on_line=on_line)


def recon(ROOT_DIR):
    if not os.path.isdir(ROOT_DIR):
        raise FileNotFoundError(f"root dir does not exist: {ROOT_DIR}")

    try:
        returncode = _recon_via_warm_server(ROOT_DIR)
        print(f"\nExit code: {returncode}")
        if returncode != 0:
            raise RuntimeError(f"recon server job exited with code {returncode}")
        return returncode
    except ConnectionRefusedError:
        pass  # server not running - fall back below
    except ImportError:
        pass  # RTKRecon/recon_client.py missing - fall back below

    print("recon server not running - falling back to a fresh subprocess "
          "(start RTKRecon/start_recon_server.bat to skip the itk/rtk warm-up next time).",
          flush=True)
    BAT_PATH = os.path.join(PIPELINE_DIR, "recon_phase.bat")
    EXTRA_ARGS = []
    cmd = [BAT_PATH, ROOT_DIR, *EXTRA_ARGS]
    process = subprocess.Popen(
        cmd,
        cwd=PIPELINE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
        sys.stdout.flush()
    returncode = process.wait()
    print(f"\nExit code: {returncode}")
    if returncode != 0:
        raise RuntimeError(f"recon_phase.bat exited with code {returncode}")
    return returncode


from register_phases import register_all_phases


def register_step(main_dir, outname, EE_toEI_only=False):
    results_dir = os.path.join(main_dir, outname, 'Results')
    registered_dir = os.path.join(results_dir, 'Registered')
    os.makedirs(registered_dir, exist_ok=True)

    only_phases = [7] if EE_toEI_only else None
    failures = register_all_phases(results_dir, only_phases=only_phases)
    if failures:
        print(f"Failed phases: {failures}")
    return failures


def get_difformation_fields(main_dir, outname,
                             PHASE=7,
                             ARROW_SIZE_SCALE=6,
                             MIN_VOXELS_PER_ARROW=6):
    ARROW_BLOCK = 25
    COLOR_MIN_MM = 0
    COLOR_MAX_MM = 6
    DENSE_STRIDE = 2

    RESULTS_DIR = os.path.join(main_dir, outname, "Results")
    Registered_dir = os.path.join(RESULTS_DIR, "Registered")
    if not os.path.isdir(Registered_dir) or not os.listdir(Registered_dir):
        raise FileNotFoundError(f"No registered phases found in {Registered_dir} yet - run register_step first.")

    candidate_dirs = sorted(
        d for d in os.listdir(Registered_dir)
        if os.path.isdir(os.path.join(Registered_dir, d)) and "_to_" in d
    )
    if not candidate_dirs:
        raise FileNotFoundError(
            f"No folders matching '*_to_*' found in {Registered_dir} - run register_step first."
        )
    folder_name = candidate_dirs[0]
    fix_phase = int(folder_name.split('_to_')[-1].split('R')[-1])

    out_dir = os.path.join(Registered_dir, f"R{PHASE}_to_R{fix_phase}")
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Expected registration output folder not found: {out_dir}")

    warp_candidates = [os.path.join(out_dir, x) for x in os.listdir(out_dir) if "Forward.nii.gz" in x]
    if not warp_candidates:
        raise FileNotFoundError(f"No '*Forward.nii.gz' warp file found in {out_dir}")
    warp_path = warp_candidates[0]

    mask_candidates = [os.path.join(RESULTS_DIR, x) for x in os.listdir(RESULTS_DIR) if f"R{PHASE}_m.nii" in x]
    if not mask_candidates:
        raise FileNotFoundError(f"No mask file matching 'R{PHASE}_m.nii*' found in {RESULTS_DIR}")
    mask_path = mask_candidates[0]

    html_out = os.path.join(out_dir, "deformation_field.html")

    warp_img = nib.load(warp_path)
    mask_img = nib.load(mask_path)
    affine = warp_img.affine
    warp = warp_img.get_fdata().squeeze()
    mask = mask_img.get_fdata() != 0
    assert warp.shape[:3] == mask.shape, "warp field and mask are on different grids"

    ai, aj, ak = np.where(mask)
    if ai.size == 0:
        raise ValueError(f"Mask {mask_path} is empty - no lung voxels found.")

    block_id = np.stack([ai // ARROW_BLOCK, aj // ARROW_BLOCK, ak // ARROW_BLOCK], axis=1)
    uniq_blocks, inverse = np.unique(block_id, axis=0, return_inverse=True)
    vecs_all = warp[ai, aj, ak, :]
    n_blocks = len(uniq_blocks)
    vec_sums = np.zeros((n_blocks, 3))
    idx_sums = np.zeros((n_blocks, 3))
    counts = np.zeros(n_blocks)
    np.add.at(vec_sums, inverse, vecs_all)
    np.add.at(idx_sums, inverse, np.stack([ai, aj, ak], axis=1).astype(float))
    np.add.at(counts, inverse, 1)
    keep_blocks = counts >= MIN_VOXELS_PER_ARROW
    mean_vecs = vec_sums[keep_blocks] / counts[keep_blocks, None]
    mean_idx = idx_sums[keep_blocks] / counts[keep_blocks, None]

    if len(mean_idx) == 0:
        raise ValueError(
            f"No blocks passed MIN_VOXELS_PER_ARROW={MIN_VOXELS_PER_ARROW}; "
            "lower this threshold or check the mask."
        )

    print(f"averaged arrows: {len(mean_idx):,} (out of {n_blocks:,} candidate blocks)")
    arrow_vox = np.concatenate([mean_idx, np.ones((len(mean_idx), 1))], axis=1)
    arrow_world = (affine @ arrow_vox.T).T[:, :3]

    # Uniform stride over the flat voxel list, not a per-axis parity AND -
    # see the note in _diaphragm_2d_arrays for why the AND scheme can
    # silently drop an entire spatial region to zero points.
    di, dj, dk = ai, aj, ak
    if DENSE_STRIDE > 1:
        stride_n = max(int(DENSE_STRIDE) ** 3, 1)
        di, dj, dk = di[::stride_n], dj[::stride_n], dk[::stride_n]
    dense_vox = np.stack([di, dj, dk, np.ones_like(di)], axis=1).astype(float)
    dense_world = (affine @ dense_vox.T).T[:, :3]
    print(f"scatter points: {len(di):,}")

    fig_new = go.Figure()
    fig_new.add_trace(go.Scatter3d(
        x=dense_world[:, 0], y=dense_world[:, 1], z=dense_world[:, 2],
        mode="markers",
        marker=dict(size=2, color="gray", opacity=0.4),
        name="lung"))
    fig_new.add_trace(go.Cone(
        x=arrow_world[:, 0], y=arrow_world[:, 1], z=arrow_world[:, 2],
        u=mean_vecs[:, 0], v=mean_vecs[:, 1], w=mean_vecs[:, 2],
        colorscale="jet", sizemode="scaled", sizeref=ARROW_SIZE_SCALE,
        cmin=COLOR_MIN_MM, cmax=COLOR_MAX_MM,
        colorbar=dict(title="mm", x=0.85),
        name="region-averaged direction"))

    DARK_BG = "black"
    SHOW_AXES = False
    AXIS_STYLE = dict(backgroundcolor=DARK_BG, gridcolor="dimgray", zerolinecolor="dimgray",
                       color="white", visible=SHOW_AXES)
    fig_new.update_layout(
        scene=dict(aspectmode="data",
                   xaxis=dict(title="X (mm)", **AXIS_STYLE),
                   yaxis=dict(title="Y (mm)", **AXIS_STYLE),
                   zaxis=dict(title="Z (mm)", **AXIS_STYLE),
                   bgcolor=DARK_BG),
        paper_bgcolor=DARK_BG,
        font=dict(color="white"),
        title=f"R{PHASE}\u2192R{fix_phase} displacement magnitude with region-averaged direction arrows",
        width=1000, height=900)
    fig_new.update_traces(
        colorbar=dict(title=dict(text="mm", font=dict(color="white")), tickfont=dict(color="white")),
        selector=dict(type="cone"))
    fig_new.write_html(html_out)
    print("standalone interactive copy saved to:", html_out)
    log_artifact(html_out)


# --------------------------------------------------------------------------
# Diaphragm motion estimate - copied as-is (headless) from
# Desktop/Pipeline/estimate_diaphragm_motion.ipynb, so behavior matches the
# notebook exactly. Requires the "register" step (EE_toEI_only, PHASE=7) to
# have run first - raises FileNotFoundError with a clear message otherwise.
# --------------------------------------------------------------------------
def _locate_diaphragm_warp_and_mask(main_dir, outname, PHASE):
    RESULTS_DIR = os.path.join(main_dir, outname, "Results")
    Registered_dir = os.path.join(RESULTS_DIR, "Registered")
    if not os.path.isdir(Registered_dir) or not os.listdir(Registered_dir):
        raise FileNotFoundError(f"No registration found in {Registered_dir} - run the Register step first.")

    candidate_dirs = sorted(
        d for d in os.listdir(Registered_dir)
        if os.path.isdir(os.path.join(Registered_dir, d)) and "_to_" in d
    )
    if not candidate_dirs:
        raise FileNotFoundError(f"No folders matching '*_to_*' found in {Registered_dir} - run the Register step first.")
    folder_name = candidate_dirs[0]
    fix_phase = int(folder_name.split('_to_')[-1].split('R')[-1])

    out_dir = os.path.join(Registered_dir, f"R{PHASE}_to_R{fix_phase}")
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Expected registration output folder not found: {out_dir} - run the Register step first.")

    warp_candidates = [os.path.join(out_dir, x) for x in os.listdir(out_dir) if "Forward.nii.gz" in x]
    if not warp_candidates:
        raise FileNotFoundError(f"No '*Forward.nii.gz' warp file found in {out_dir} - run the Register step first.")
    warp_path = warp_candidates[0]

    mask_candidates = [os.path.join(RESULTS_DIR, x) for x in os.listdir(RESULTS_DIR) if f"R{PHASE}_m.nii" in x]
    if not mask_candidates:
        raise FileNotFoundError(f"No mask file matching 'R{PHASE}_m.nii*' found in {RESULTS_DIR}")
    mask_path = mask_candidates[0]

    return warp_path, mask_path, out_dir, fix_phase


def _find_diaphragm_lr_split_index(ai, x_size, search_band=(0.3, 0.7)):
    counts = np.bincount(ai, minlength=x_size)
    nz = np.where(counts > 0)[0]
    lo, hi = nz.min(), nz.max()
    span = hi - lo
    band_lo = lo + int(search_band[0] * span)
    band_hi = lo + int(search_band[1] * span)
    band = counts[band_lo:band_hi + 1]
    split_idx = band_lo + int(np.argmin(band))
    if band.min() > 0.5 * band.max():
        print(f"WARNING: no clear mediastinal gap found (valley count {band.min()} vs "
              f"band max {band.max()}) - left/right split may be unreliable, verify visually.")
    return split_idx


def _split_diaphragm_left_right(ai, x_size, affine):
    split_idx = _find_diaphragm_lr_split_index(ai, x_size)
    axis0_code = nib.aff2axcodes(affine)[0]  # 'R' or 'L': which way +voxel-index-0 points per the header
    higher_index_is_right = (axis0_code != "R")  # inverted: header-literal mapping was confirmed backwards
    is_right = (ai > split_idx) if higher_index_is_right else (ai < split_idx)
    is_left = ~is_right
    return is_left, is_right, split_idx


_DIAPHRAGM_REGION_CACHE = {}


def _compute_diaphragm_region(main_dir, outname, PHASE=7, diaphragm_fraction=0.15, side="both"):
    # The warp field alone is typically a multi-GB gzipped NIfTI file -
    # ~24s to load and decompress on this pipeline's data - and a single
    # diaphragm_step() run calls this function 10 times across
    # get_diaphragm_descent_mm/_lr, visualize_diaphragm_descent, and the
    # three 2D-view/panel functions, 6 of those for side="both" alone, even
    # though the result only depends on (main_dir, outname, PHASE,
    # diaphragm_fraction, side) and nothing changes between those calls
    # within one run. Caching collapses that down to 3 real loads (one per
    # side actually used). Each caller gets a shallow copy of the cached
    # dict - the large numpy arrays (warp, mask, ...) aren't copied, only
    # the dict container - so a caller that stashes extra keys onto its
    # copy (visualize_diaphragm_descent adds "figure"/"html_out") can't
    # pollute the shared cache entry or leak into another caller's copy.
    cache_key = (os.path.normcase(os.path.normpath(main_dir)), outname, PHASE, diaphragm_fraction, side)
    cached = _DIAPHRAGM_REGION_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    if side not in ("both", "left", "right"):
        raise ValueError(f"side must be 'both', 'left', or 'right', got {side!r}")

    warp_path, mask_path, out_dir, fix_phase = _locate_diaphragm_warp_and_mask(main_dir, outname, PHASE)

    warp_img = nib.load(warp_path)
    mask_img = nib.load(mask_path)
    affine = warp_img.affine
    warp = np.asarray(warp_img.dataobj).squeeze()
    mask = np.asarray(mask_img.dataobj) != 0
    if warp.shape[:3] != mask.shape:
        raise ValueError("warp field and mask are on different grids")

    ai, aj, ak = np.where(mask)
    if ai.size == 0:
        raise ValueError(f"Mask {mask_path} is empty - no lung voxels found.")

    split_idx = None
    if side != "both":
        is_left, is_right, split_idx = _split_diaphragm_left_right(ai, mask.shape[0], affine)
        keep = is_right if side == "right" else is_left
        ai, aj, ak = ai[keep], aj[keep], ak[keep]
        if ai.size == 0:
            raise ValueError(f"No voxels ended up on the '{side}' side of the L/R split "
                              f"(split_idx={split_idx}) - check the mask / split_idx.")

    vox = np.stack([ai, aj, ak, np.ones_like(ai)], axis=1).astype(float)
    world = (affine @ vox.T).T[:, :3]
    z_world = world[:, 2]  # world RAS Z: +Z = superior/up, -Z = inferior/down

    z_min, z_max = z_world.min(), z_world.max()
    z_thresh = z_min + diaphragm_fraction * (z_max - z_min)
    in_diaphragm = z_world <= z_thresh
    if not np.any(in_diaphragm):
        raise ValueError("No voxels fell inside the diaphragm region - check diaphragm_fraction.")

    dz_region = warp[ai[in_diaphragm], aj[in_diaphragm], ak[in_diaphragm], 2]
    mean_dz = float(dz_region.mean())
    descent_mm = -mean_dz  # positive number = moved downward

    result = dict(
        descent_mm=descent_mm,
        mean_dz=mean_dz,
        std_dz=float(dz_region.std()),
        n_voxels=int(in_diaphragm.sum()),
        PHASE=PHASE, fix_phase=fix_phase, out_dir=out_dir,
        affine=affine, warp=warp, mask=mask,
        ai=ai, aj=aj, ak=ak, world=world, z_world=z_world,
        in_diaphragm=in_diaphragm, z_thresh=z_thresh,
        diaphragm_fraction=diaphragm_fraction,
        side=side, split_voxel_index=split_idx,
    )
    _DIAPHRAGM_REGION_CACHE[cache_key] = result
    return dict(result)


def get_diaphragm_descent_mm(main_dir, outname, PHASE=7, diaphragm_fraction=0.15, side="both"):
    data = _compute_diaphragm_region(main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction, side=side)
    return data["descent_mm"]


def get_diaphragm_descent_mm_lr(main_dir, outname, PHASE=7, diaphragm_fraction=0.15):
    return {
        "left": get_diaphragm_descent_mm(main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction, side="left"),
        "right": get_diaphragm_descent_mm(main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction, side="right"),
    }


def visualize_diaphragm_descent(main_dir, outname, PHASE=7, diaphragm_fraction=0.15, side="both",
                                 ARROW_BLOCK=15, MIN_VOXELS_PER_ARROW=4,
                                 DENSE_STRIDE=2, html_out=None):
    data = _compute_diaphragm_region(main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction, side=side)
    warp, mask, affine = data["warp"], data["mask"], data["affine"]
    ai, aj, ak = data["ai"], data["aj"], data["ak"]  # already filtered to `side`
    world, in_diaphragm = data["world"], data["in_diaphragm"]

    if html_out is None:
        suffix = "" if side == "both" else f"_{side}"
        html_out = os.path.join(data["out_dir"], f"diaphragm_descent{suffix}.html")

    # Uniform stride over the flat voxel list, not a per-axis parity AND -
    # see the note in _diaphragm_2d_arrays for why the AND scheme can
    # silently drop an entire spatial region to zero points.
    stride_n = max(int(DENSE_STRIDE) ** 3, 1)

    other_world = None
    if side != "both":
        all_ai, all_aj, all_ak = np.where(mask)
        is_left, is_right, _ = _split_diaphragm_left_right(all_ai, mask.shape[0], affine)
        other_sel = is_left if side == "right" else is_right
        oi, oj, ok = all_ai[other_sel], all_aj[other_sel], all_ak[other_sel]
        if stride_n > 1:
            oi, oj, ok = oi[::stride_n], oj[::stride_n], ok[::stride_n]
        other_vox = np.stack([oi, oj, ok, np.ones_like(oi)], axis=1).astype(float)
        other_world = (affine @ other_vox.T).T[:, :3]

    rest = ~in_diaphragm
    ri, rj, rk = ai[rest], aj[rest], ak[rest]
    if stride_n > 1:
        ri, rj, rk = ri[::stride_n], rj[::stride_n], rk[::stride_n]
    rest_vox = np.stack([ri, rj, rk, np.ones_like(ri)], axis=1).astype(float)
    rest_world = (affine @ rest_vox.T).T[:, :3]

    di_idx = np.where(in_diaphragm)[0]
    di_i, di_j, di_k = ai[di_idx], aj[di_idx], ak[di_idx]
    if stride_n > 1:
        di_i, di_j, di_k = di_i[::stride_n], di_j[::stride_n], di_k[::stride_n]
    diaphragm_vox = np.stack([di_i, di_j, di_k, np.ones_like(di_i)], axis=1).astype(float)
    diaphragm_world = (affine @ diaphragm_vox.T).T[:, :3]

    bi, bj, bk = ai[in_diaphragm], aj[in_diaphragm], ak[in_diaphragm]
    block_id = np.stack([bi // ARROW_BLOCK, bj // ARROW_BLOCK, bk // ARROW_BLOCK], axis=1)
    uniq_blocks, inverse = np.unique(block_id, axis=0, return_inverse=True)
    vecs_all = warp[bi, bj, bk, :]
    n_blocks = len(uniq_blocks)
    vec_sums = np.zeros((n_blocks, 3))
    idx_sums = np.zeros((n_blocks, 3))
    counts = np.zeros(n_blocks)
    np.add.at(vec_sums, inverse, vecs_all)
    np.add.at(idx_sums, inverse, np.stack([bi, bj, bk], axis=1).astype(float))
    np.add.at(counts, inverse, 1)
    keep_blocks = counts >= MIN_VOXELS_PER_ARROW
    if keep_blocks.sum() == 0:
        keep_blocks = counts >= 1
    mean_vecs = vec_sums[keep_blocks] / counts[keep_blocks, None]
    mean_idx = idx_sums[keep_blocks] / counts[keep_blocks, None]
    arrow_vox = np.concatenate([mean_idx, np.ones((len(mean_idx), 1))], axis=1)
    arrow_world = (affine @ arrow_vox.T).T[:, :3]

    ref_origin = diaphragm_world.mean(axis=0)
    ref_origin[0] += (world[:, 0].max() - world[:, 0].min()) * 0.35
    ref_length = max((data["z_world"].max() - data["z_world"].min()) * 0.25, 1.0)

    CONE_HOVERTEMPLATE = (
        "x=%{x:.2f} mm  y=%{y:.2f} mm  z=%{z:.2f} mm<br>"
        "dx=%{u:.3f} mm  dy=%{v:.3f} mm  dz=%{w:.3f} mm<br>"
        "norm=%{norm:.3f} mm<extra></extra>"
    )

    fig = go.Figure()
    if other_world is not None:
        fig.add_trace(go.Scatter3d(
            x=other_world[:, 0], y=other_world[:, 1], z=other_world[:, 2],
            mode="markers", marker=dict(size=2, color="dimgray", opacity=0.15),
            name="other lung (not selected)"))
    fig.add_trace(go.Scatter3d(
        x=rest_world[:, 0], y=rest_world[:, 1], z=rest_world[:, 2],
        mode="markers",
        marker=dict(size=2, color=("steelblue" if side != "both" else "gray"), opacity=0.3),
        name=("rest of selected lung" if side != "both" else "rest of lung")))
    fig.add_trace(go.Scatter3d(
        x=diaphragm_world[:, 0], y=diaphragm_world[:, 1], z=diaphragm_world[:, 2],
        mode="markers", marker=dict(size=2.5, color="orange", opacity=0.55),
        name=f"diaphragm region (bottom {diaphragm_fraction:.0%})"))
    fig.add_trace(go.Cone(
        x=arrow_world[:, 0], y=arrow_world[:, 1], z=arrow_world[:, 2],
        u=mean_vecs[:, 0], v=mean_vecs[:, 1], w=mean_vecs[:, 2],
        colorscale="jet", sizemode="scaled", sizeref=4,
        colorbar=dict(title="mm", x=0.85),
        hovertemplate=CONE_HOVERTEMPLATE,
        name="region-averaged displacement"))
    fig.add_trace(go.Cone(
        x=[ref_origin[0]], y=[ref_origin[1]], z=[ref_origin[2]],
        u=[0], v=[0], w=[-ref_length],
        sizemode="absolute", sizeref=ref_length * 0.5,
        colorscale=[[0, "black"], [1, "black"]], showscale=False,
        hovertemplate="reference: DOWN (-Z / inferior)<br>fixed length for visibility, not real data<extra></extra>",
        name="reference: DOWN (-Z / inferior)"))

    DARK_BG = "black"
    side_label = {"both": "both lungs", "left": "left lung", "right": "right lung"}[side]
    fig.update_layout(
        scene=dict(aspectmode="data",
                   xaxis=dict(title="X (mm)", backgroundcolor=DARK_BG, color="white", visible=False),
                   yaxis=dict(title="Y (mm)", backgroundcolor=DARK_BG, color="white", visible=False),
                   zaxis=dict(title="Z (mm, + = superior/up)", backgroundcolor=DARK_BG, color="white", visible=True),
                   bgcolor=DARK_BG),
        paper_bgcolor=DARK_BG,
        font=dict(color="white"),
        title=(f"R{PHASE}→R{data['fix_phase']} {side_label} diaphragm descent = {data['descent_mm']:.3f} mm "
               f"(n={data['n_voxels']:,} voxels, black cone = true DOWN direction)"),
        width=1100, height=950,
        legend=dict(font=dict(color="white")),
    )
    fig.write_html(html_out)
    print(f"{side_label} diaphragm descent: {data['descent_mm']:.3f} mm (mean dz={data['mean_dz']:.3f}, "
          f"std={data['std_dz']:.3f}, n={data['n_voxels']:,} voxels)")
    print("interactive visualization saved to:", html_out)

    data["figure"] = fig
    data["html_out"] = html_out
    return data


# Set to True to re-enable the rotating-camera GIF in diaphragm_step().
# Paused since it's the single slowest part of the step (36 kaleido-rendered
# frames, ~5 minutes) and unrelated to the redundant-computation fix - every
# other output (HTML plots, the three 2D views, the panel, the analysis
# txt) still gets generated normally while this is off.
GENERATE_DIAPHRAGM_GIF = False


def _save_diaphragm_rotation_gif(fig, gif_out, n_frames=36, camera_radius=2.0,
                                  camera_elevation=0.7, frame_duration_ms=100):
    """Rotating-camera GIF export of the diaphragm plotly figure - same
    approach as visualize_deformation_field.ipynb's 'Rotating GIF export'
    cell (requires kaleido for static rendering)."""
    import io
    from PIL import Image as PILImage
    frames = []
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        fig.update_layout(scene_camera=dict(
            eye=dict(x=camera_radius * np.cos(theta), y=camera_radius * np.sin(theta), z=camera_elevation)
        ))
        png_bytes = fig.to_image(format="png", scale=1)
        frames.append(PILImage.open(io.BytesIO(png_bytes)).convert("RGB"))
    frames[0].save(gif_out, save_all=True, append_images=frames[1:], duration=frame_duration_ms, loop=0)
    return gif_out


PLANE_CONFIG = {
    # h/v = which world axis (0=X/L-R, 1=Y/A-P, 2=Z/S-I) is horizontal/
    # vertical on screen for each plane; everything in
    # visualize_diaphragm_descent_2d below this is generic across all three.
    "coronal":  dict(h=0, v=2, h_label="X (mm, left/right)", v_label="Z (mm, superior/inferior)",
                      desc="coronal projection (Y/anterior-posterior collapsed)"),
    "sagittal": dict(h=1, v=2, h_label="Y (mm, anterior/posterior)", v_label="Z (mm, superior/inferior)",
                      desc="sagittal projection (X/left-right collapsed)"),
    "axial":    dict(h=0, v=1, h_label="X (mm, left/right)", v_label="Y (mm, anterior/posterior)",
                      desc="axial projection (Z/superior-inferior collapsed)"),
}


def _diaphragm_2d_arrays(main_dir, outname, PHASE=7, diaphragm_fraction=0.15, side="both",
                          ARROW_BLOCK=15, MIN_VOXELS_PER_ARROW=4, DENSE_STRIDE=2):
    """
    Everything the 2D diaphragm visualizations need that does NOT depend on
    which anatomical plane is being drawn: the warp/mask data, the
    diaphragm-region and "rest of lung" point clouds (already thinned by
    DENSE_STRIDE), the block-averaged arrow vectors/positions/magnitudes -
    covering the WHOLE lung, not just the diaphragm region; only the white
    highlight and the reported descent number are restricted to the bottom
    `diaphragm_fraction` - and the Left/Right + Anterior/Posterior/Superior/
    Inferior orientation facts needed for edge labels. Computed once and
    shared across all three plane renders (coronal/sagittal/axial) instead
    of redone per plane - used by both visualize_diaphragm_descent_2d (one
    plane) and visualize_diaphragm_descent_2d_panel (all three, one shared
    colorbar).
    """
    data = _compute_diaphragm_region(main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction, side=side)
    warp, mask, affine = data["warp"], data["mask"], data["affine"]
    ai, aj, ak = data["ai"], data["aj"], data["ak"]  # already filtered to `side`
    in_diaphragm = data["in_diaphragm"]

    # Thinning for the background scatter clouds below: NOT a per-axis
    # "index % DENSE_STRIDE == 0" AND across i/j/k. That scheme requires all
    # three voxel indices to be simultaneously divisible by DENSE_STRIDE, and
    # if a whole anatomical region happens to sit on the "wrong" parity in
    # even one axis (confirmed on real data - a curving lower-lung region
    # where every voxel had an odd index along one axis), it silently drops
    # to ZERO points there, regardless of how many thousands of real mask
    # voxels exist there - not sparser, completely empty, reading as a gap or
    # a disconnected floating region even though the lung is one solid,
    # connected shape. A uniform stride over the flat voxel list (every Nth
    # voxel as returned by np.where, keeping matched i/j/k triples aligned)
    # can't fail that way: it doesn't depend on any voxel's spatial index
    # value, so it can't blank out a whole region. N is chosen to keep
    # roughly the same point density as the old scheme (~1/DENSE_STRIDE^3).
    stride_n = max(int(DENSE_STRIDE) ** 3, 1)

    other_world = None
    if side != "both":
        all_ai, all_aj, all_ak = np.where(mask)
        is_left, is_right, _ = _split_diaphragm_left_right(all_ai, mask.shape[0], affine)
        other_sel = is_left if side == "right" else is_right
        oi, oj, ok = all_ai[other_sel], all_aj[other_sel], all_ak[other_sel]
        if stride_n > 1:
            oi, oj, ok = oi[::stride_n], oj[::stride_n], ok[::stride_n]
        other_vox = np.stack([oi, oj, ok, np.ones_like(oi)], axis=1).astype(float)
        other_world = (affine @ other_vox.T).T[:, :3]

    rest = ~in_diaphragm
    ri, rj, rk = ai[rest], aj[rest], ak[rest]
    if stride_n > 1:
        ri, rj, rk = ri[::stride_n], rj[::stride_n], rk[::stride_n]
    rest_vox = np.stack([ri, rj, rk, np.ones_like(ri)], axis=1).astype(float)
    rest_world = (affine @ rest_vox.T).T[:, :3]

    di_idx = np.where(in_diaphragm)[0]
    di_i, di_j, di_k = ai[di_idx], aj[di_idx], ak[di_idx]
    if stride_n > 1:
        di_i, di_j, di_k = di_i[::stride_n], di_j[::stride_n], di_k[::stride_n]
    diaphragm_vox = np.stack([di_i, di_j, di_k, np.ones_like(di_i)], axis=1).astype(float)
    diaphragm_world = (affine @ diaphragm_vox.T).T[:, :3]

    # Arrows are block-averaged over the WHOLE lung (ai/aj/ak, every voxel
    # for this `side`), not just the in_diaphragm subset - the diaphragm
    # region (white highlight above, and the reported descent number) stays
    # restricted to the bottom `diaphragm_fraction`, but the arrows
    # themselves show the full deformation field across the entire lung,
    # same as the whole-lung arrows in the Register step's
    # deformation_field.html (get_difformation_fields).
    bi, bj, bk = ai, aj, ak
    block_id = np.stack([bi // ARROW_BLOCK, bj // ARROW_BLOCK, bk // ARROW_BLOCK], axis=1)
    uniq_blocks, inverse = np.unique(block_id, axis=0, return_inverse=True)
    vecs_all = warp[bi, bj, bk, :]
    n_blocks = len(uniq_blocks)
    vec_sums = np.zeros((n_blocks, 3))
    idx_sums = np.zeros((n_blocks, 3))
    counts = np.zeros(n_blocks)
    np.add.at(vec_sums, inverse, vecs_all)
    np.add.at(idx_sums, inverse, np.stack([bi, bj, bk], axis=1).astype(float))
    np.add.at(counts, inverse, 1)
    keep_blocks = counts >= MIN_VOXELS_PER_ARROW
    if keep_blocks.sum() == 0:
        keep_blocks = counts >= 1
    mean_vecs = vec_sums[keep_blocks] / counts[keep_blocks, None]
    mean_idx = idx_sums[keep_blocks] / counts[keep_blocks, None]
    arrow_vox = np.concatenate([mean_idx, np.ones((len(mean_idx), 1))], axis=1)
    arrow_world = (affine @ arrow_vox.T).T[:, :3]
    arrow_norm = np.linalg.norm(mean_vecs, axis=1)  # full 3D magnitude, used for arrow color

    # Which screen edge (higher vs. lower world value) is which anatomical
    # side - computed from the actual mask geometry rather than assumed.
    # Left/Right (world axis 0) needs the empirically-corrected split (see
    # _split_diaphragm_left_right - the affine's own axis code was found
    # backwards for this axis); so does Anterior/Posterior (axis 1) -
    # reading it naively off the affine's own code put A and P on the wrong
    # sides too. Superior/Inferior (axis 2) hasn't shown that problem, so
    # it alone is read directly from the affine's own RAS axis code.
    all_ai0, all_aj0, all_ak0 = np.where(mask)
    is_left_all, is_right_all, _ = _split_diaphragm_left_right(all_ai0, mask.shape[0], affine)
    all_vox0 = np.stack([all_ai0, all_aj0, all_ak0, np.ones_like(all_ai0)], axis=1).astype(float)
    all_world_x = (affine @ all_vox0.T).T[:, 0]
    left_mean_x = all_world_x[is_left_all].mean() if is_left_all.any() else 0.0
    right_mean_x = all_world_x[is_right_all].mean() if is_right_all.any() else 0.0
    higher_x_is_left = left_mean_x > right_mean_x
    axcodes = nib.aff2axcodes(affine)  # e.g. ('R', 'A', 'S') - code for each axis' *positive* direction

    return dict(data=data, other_world=other_world, rest_world=rest_world,
                diaphragm_world=diaphragm_world, arrow_world=arrow_world,
                mean_vecs=mean_vecs, arrow_norm=arrow_norm,
                higher_x_is_left=higher_x_is_left, axcodes=axcodes)


def _render_diaphragm_plane(ax, arrays, plane, side, diaphragm_fraction, ARROW_SCALE=1.0,
                             vmin=0.0, vmax=5.0, show_legend=True, show_diaphragm_region=False):
    """
    Draws one plane's projection (coronal/sagittal/axial) of the prepared
    `arrays` (from _diaphragm_2d_arrays) onto `ax`: the lung/diaphragm point
    clouds, the true-to-scale block-averaged displacement arrows (colored
    by magnitude, clamped to [vmin, vmax] so color is directly comparable
    across planes and across the standalone/panel outputs instead of each
    auto-scaling to its own data range), edge orientation labels, axis
    labels, and a top-anchored equal-aspect FOV sized to include every
    arrow tip. Returns the quiver mappable (for an external colorbar).
    """
    cfg = PLANE_CONFIG[plane]
    h_axis, v_axis = cfg["h"], cfg["v"]
    other_world = arrays["other_world"]
    rest_world = arrays["rest_world"]
    diaphragm_world = arrays["diaphragm_world"]
    arrow_world = arrays["arrow_world"]
    mean_vecs = arrays["mean_vecs"]
    arrow_norm = arrays["arrow_norm"]
    higher_x_is_left = arrays["higher_x_is_left"]
    axcodes = arrays["axcodes"]

    def _hv(pts):
        return pts[:, h_axis], pts[:, v_axis]

    # True-to-scale arrows: 1 mm of real displacement is drawn as 1 mm on
    # these mm-labeled axes, same as the lung silhouette itself. No
    # auto-fit-to-cell scaling and no overshoot capping - ARROW_SCALE
    # (default 1.0) is the only multiplier, for a deliberate manual
    # exaggeration/reduction if a dataset ever needs one; it does not
    # change the reported descent numbers, only how long the arrows are
    # drawn. At 1.0, some arrows may legitimately cross their neighbors
    # when real displacement exceeds the ARROW_BLOCK cell spacing - that's
    # the actual data, not a rendering artifact to hide.
    u = mean_vecs[:, h_axis] * ARROW_SCALE
    v = mean_vecs[:, v_axis] * ARROW_SCALE

    def _edge_labels(axis_idx):
        """(label at the low/negative end, label at the high/positive end) of a world axis."""
        if axis_idx == 0:
            return ("R", "L") if higher_x_is_left else ("L", "R")
        if axis_idx == 1:
            return ("A", "P") if axcodes[1] == "A" else ("P", "A")
        return ("I", "S") if axcodes[2] == "S" else ("S", "I")

    low_h_label, high_h_label = _edge_labels(h_axis)
    low_v_label, high_v_label = _edge_labels(v_axis)

    DARK_BG = "black"
    ax.set_facecolor(DARK_BG)

    if other_world is not None:
        ox, oz = _hv(other_world)
        ax.scatter(ox, oz, s=4, c="dimgray", alpha=0.15, label="other lung (not selected)")

    rx, rz = _hv(rest_world)
    ax.scatter(rx, rz, s=4, c=("steelblue" if side != "both" else "gray"), alpha=0.3,
               label=("rest of selected lung" if side != "both" else "rest of lung"))

    # The diaphragm-region voxels (bottom `diaphragm_fraction` of the lung)
    # need the same baseline gray context dots as the rest of the lung -
    # arrows cover the whole lung including this region (see
    # _diaphragm_2d_arrays), so skipping their background dots here left
    # those arrows with no surrounding mask context at all, reading as if
    # they floated in a void. Always draw them in the same base color, and
    # only ADD the white highlight on top when show_diaphragm_region is True.
    dpx, dpz = _hv(diaphragm_world)
    ax.scatter(dpx, dpz, s=4, c=("steelblue" if side != "both" else "gray"), alpha=0.3)
    if show_diaphragm_region:
        ax.scatter(dpx, dpz, s=6, c="white", alpha=0.55,
                   label=f"diaphragm region (bottom {diaphragm_fraction:.0%})")

    aax, aaz = _hv(arrow_world)
    q = ax.quiver(aax, aaz, u, v, arrow_norm, cmap="jet", angles="xy", scale_units="xy", scale=1,
                  width=0.012, headwidth=4.5, headlength=5.5, headaxislength=5, zorder=5)
    q.set_clim(vmin, vmax)  # Quiver doesn't take vmin/vmax as constructor kwargs

    # No separate "DOWN" / scale-key reference inset: arrows are drawn true
    # to scale, so the mm axes themselves are the only length calibration
    # needed - but which screen edge is which side still needs to be
    # stated explicitly, since +/- axis values alone don't tell a reader
    # which is which. Edge labels below answer that directly, in
    # axes-fraction coordinates so they sit at the plot frame regardless of
    # the data's actual mm range.
    label_kw = dict(transform=ax.transAxes, color="white", fontsize=13, fontweight="bold", zorder=10)
    ax.text(0.012, 0.5, low_h_label, ha="left", va="center", **label_kw)
    ax.text(0.988, 0.5, high_h_label, ha="right", va="center", **label_kw)
    ax.text(0.5, 0.985, high_v_label, ha="center", va="top", **label_kw)
    ax.text(0.5, 0.015, low_v_label, ha="center", va="bottom", **label_kw)

    ax.set_xlabel(cfg["h_label"], color="white")
    ax.set_ylabel(cfg["v_label"], color="white")

    # FOV must cover every arrow *tip*, not just the point cloud - a long
    # arrow starting near the bottom/side edge of the lung would otherwise
    # get clipped by the axes limits (matplotlib autoscales to the scatter
    # data and the arrow *tails*, but tails+vectors can extend past that).
    tip_x, tip_z = aax + u, aaz + v
    xs_all = np.concatenate([rx, dpx, aax, tip_x] + ([ox] if other_world is not None else []))
    zs_all = np.concatenate([rz, dpz, aaz, tip_z] + ([oz] if other_world is not None else []))
    x_span = xs_all.max() - xs_all.min()
    z_span = zs_all.max() - zs_all.min()
    x_pad = max(0.08 * x_span, 1.0)
    z_pad = max(0.08 * z_span, 1.0)
    ax.set_xlim(xs_all.min() - x_pad, xs_all.max() + x_pad)
    ax.set_ylim(zs_all.min() - z_pad, zs_all.max() + z_pad)
    ax.set_aspect("equal", adjustable="box")
    # Anchor the (now aspect-shrunk) box to the top instead of the default
    # center: a wide-but-short plane like axial shrinks ax's height to
    # match its data aspect, and a centered box leaves dead space both
    # above (between the title and ax) and below it - bbox_inches="tight"
    # on savefig only crops outer margins, not that internal gap, so
    # top-anchoring pushes all the slack below ax where cropping removes it.
    ax.set_anchor("N")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    if show_legend:
        # Below the plot, not "upper left": an in-axes legend's width is a
        # fixed font size, but ax's own box width varies a lot by plane/
        # dataset (a tall-and-narrow sagittal projection shrinks ax's width
        # far more than coronal/axial do to keep the 1mm=1mm aspect) - so a
        # legend anchored inside a narrow ax can extend past the horizontal
        # center and collide with the S/A top edge label. Anchoring below
        # the axes entirely is collision-free regardless of how narrow ax
        # ends up.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=14,
                  facecolor="black", edgecolor="white", labelcolor="white")

    return q


def _diaphragm_2d_title(data, PHASE, side, descent_left, descent_right, ARROW_SCALE, vmin, vmax, desc):
    """Shared title-building logic for both the single-plane and panel figures."""
    side_label = {"both": "both lungs", "left": "left lung", "right": "right lung"}[side]
    if side == "both" and descent_left is not None and descent_right is not None:
        descent_line = f"R{PHASE}→R{data['fix_phase']} diaphragm descent - left {descent_left:.3f} mm, right {descent_right:.3f} mm"
    else:
        descent_line = f"R{PHASE}→R{data['fix_phase']} {side_label} diaphragm descent = {data['descent_mm']:.3f} mm"
    scale_note = "arrows to scale" if ARROW_SCALE == 1.0 else f"arrows drawn at {ARROW_SCALE:.2f}× true length"
    return (f"{descent_line}\n"
            f"{desc}, n={data['n_voxels']:,} voxels - "
            f"{scale_note}, color = magnitude {vmin:g}-{vmax:g} mm")


def visualize_diaphragm_descent_2d(main_dir, outname, PHASE=7, diaphragm_fraction=0.15, side="both",
                                    plane="coronal", ARROW_BLOCK=15, MIN_VOXELS_PER_ARROW=4,
                                    DENSE_STRIDE=2, ARROW_SCALE=1.0, jpg_out=None,
                                    descent_left=None, descent_right=None,
                                    vmin=0.0, vmax=5.0, show_diaphragm_region=False):
    """
    Static 2D projection of the same diaphragm-descent picture as
    visualize_diaphragm_descent(): every point and every displacement arrow
    is collapsed onto one anatomical plane by dropping the world coordinate
    perpendicular to it - a 3D-to-2D projection (like a coronal/sagittal/
    axial X-ray), not a single anatomical slice. Saved as a static
    dark-background JPG instead of the interactive HTML/GIF.

    plane="coronal" (default: X vs Z, drops Y/anterior-posterior),
    "sagittal" (Y vs Z, drops X/left-right), or "axial" (X vs Y, drops
    Z/superior-inferior).

    side="both"/"left"/"right", same meaning as visualize_diaphragm_descent.
    When side="both" and descent_left/descent_right are both given (diaphragm_step
    passes its already-computed get_diaphragm_descent_mm_lr() values), the title
    reports the per-lung descent instead of the both-lungs combined average.
    Arrows are drawn true to scale (1 mm of real displacement = 1 mm on
    these axes, same as the lung silhouette) unless ARROW_SCALE is set to
    something other than its default of 1.0 for a deliberate manual
    exaggeration/reduction; it does not change the reported descent
    numbers, only how long the arrows are drawn.

    vmin/vmax fix the color scale to a constant mm range (default 0-5mm)
    instead of auto-scaling to each dataset's own magnitude range, so
    color is directly comparable across separate runs and across the three
    plane files - same purpose as visualize_diaphragm_descent_2d_panel's
    single shared colorbar, just applied per standalone file too.

    show_diaphragm_region=True adds a white highlight over just the bottom
    diaphragm_fraction of the lung (where the reported descent number comes
    from) and appends "_diaphragm" to the default filename - arrows always
    cover the whole lung either way (see _diaphragm_2d_arrays), this only
    toggles the highlight.
    """
    if plane not in PLANE_CONFIG:
        raise ValueError(f"plane must be one of {sorted(PLANE_CONFIG)}, got {plane!r}")
    cfg = PLANE_CONFIG[plane]

    arrays = _diaphragm_2d_arrays(main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction,
                                   side=side, ARROW_BLOCK=ARROW_BLOCK, MIN_VOXELS_PER_ARROW=MIN_VOXELS_PER_ARROW,
                                   DENSE_STRIDE=DENSE_STRIDE)
    data = arrays["data"]

    if jpg_out is None:
        side_suffix = "" if side == "both" else f"_{side}"
        region_suffix = "_diaphragm" if show_diaphragm_region else ""
        jpg_out = os.path.join(data["out_dir"], f"diaphragm_2D_visualization_{plane}{side_suffix}{region_suffix}.jpg")

    DARK_BG = "black"
    fig, ax = plt.subplots(figsize=(9, 8), facecolor=DARK_BG)
    q = _render_diaphragm_plane(ax, arrays, plane, side, diaphragm_fraction, ARROW_SCALE=ARROW_SCALE,
                                 vmin=vmin, vmax=vmax, show_legend=True, show_diaphragm_region=show_diaphragm_region)

    cbar = fig.colorbar(q, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("mm", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # fig.suptitle, not ax.set_title: the axial view's rounder silhouette
    # makes `ax.set_aspect("equal", adjustable="box")` shrink ax's own box,
    # and an ax-level title sized to the *original* box width then overlaps
    # the colorbar (which sits outside ax, at a fixed figure position). A
    # figure-level title always spans the full figure width, regardless of
    # how ax gets reshaped.
    fig.suptitle(_diaphragm_2d_title(data, PHASE, side, descent_left, descent_right, ARROW_SCALE, vmin, vmax, cfg["desc"]),
                 color="white", fontsize=17, y=0.98)

    plt.savefig(jpg_out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    side_label = {"both": "both lungs", "left": "left lung", "right": "right lung"}[side]
    print(f"{side_label} 2D {plane} visualization saved to:", jpg_out)

    return jpg_out


def visualize_diaphragm_descent_2d_panel(main_dir, outname, PHASE=7, diaphragm_fraction=0.15, side="both",
                                          ARROW_BLOCK=15, MIN_VOXELS_PER_ARROW=4, DENSE_STRIDE=2,
                                          ARROW_SCALE=1.0, jpg_out=None,
                                          descent_left=None, descent_right=None,
                                          vmin=0.0, vmax=5.0, show_diaphragm_region=False):
    """
    All three projections (coronal, sagittal, axial) side by side in one
    figure, sharing a single colorbar fixed to [vmin, vmax] mm (same
    default range as visualize_diaphragm_descent_2d's standalone files) so
    magnitude is directly comparable across all three panels at a glance,
    rather than each one auto-scaling to its own data range. The heavy
    per-voxel computation (_diaphragm_2d_arrays) runs once here and is
    reused for all three panels instead of being redone three times.

    show_diaphragm_region=True adds a white highlight over just the bottom
    diaphragm_fraction of the lung in every panel and appends "_diaphragm"
    to the default filename - see visualize_diaphragm_descent_2d.
    """
    arrays = _diaphragm_2d_arrays(main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction,
                                   side=side, ARROW_BLOCK=ARROW_BLOCK, MIN_VOXELS_PER_ARROW=MIN_VOXELS_PER_ARROW,
                                   DENSE_STRIDE=DENSE_STRIDE)
    data = arrays["data"]

    if jpg_out is None:
        side_suffix = "" if side == "both" else f"_{side}"
        region_suffix = "_diaphragm" if show_diaphragm_region else ""
        jpg_out = os.path.join(data["out_dir"], f"diaphragm_2D_visualization_panel{side_suffix}{region_suffix}.jpg")

    DARK_BG = "black"
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor=DARK_BG)
    q = None
    for ax, plane in zip(axes, ("coronal", "sagittal", "axial")):
        q = _render_diaphragm_plane(ax, arrays, plane, side, diaphragm_fraction, ARROW_SCALE=ARROW_SCALE,
                                     vmin=vmin, vmax=vmax, show_legend=False, show_diaphragm_region=show_diaphragm_region)
        ax.set_title(plane.capitalize(), color="white", fontsize=12)
        # Override _render_diaphragm_plane's top-*center* anchor with
        # top-*left*: coronal/sagittal/axial have very different data
        # aspect ratios, so their equal-aspect boxes shrink to very
        # different widths, and centering each independently in an
        # equal-width cell stacks empty space from both sides into one
        # large gap between panels (most visible between coronal and
        # sagittal, the widest and narrowest of the three). Left-anchoring
        # abuts each box to its cell's left edge instead, so any leftover
        # space collects consistently on the right of each panel rather
        # than doubling up between panels.
        ax.set_anchor("NW")

    # One shared colorbar for all three panels, spanning their combined
    # height, instead of one per panel.
    cbar = fig.colorbar(q, ax=axes.tolist(), fraction=0.02, pad=0.015)
    cbar.set_label("mm", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # One shared legend below all three panels, instead of one per panel -
    # the labels are identical across planes (same underlying point
    # clouds), so any one panel's handles/labels apply to all three.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=14,
               facecolor="black", edgecolor="white", labelcolor="white",
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        _diaphragm_2d_title(data, PHASE, side, descent_left, descent_right, ARROW_SCALE, vmin, vmax,
                             "coronal / sagittal / axial projections"),
        color="white", fontsize=18, y=1.0)

    plt.savefig(jpg_out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    side_label = {"both": "both lungs", "left": "left lung", "right": "right lung"}[side]
    print(f"{side_label} 2D panel visualization saved to:", jpg_out)

    return jpg_out


def diaphragm_step(main_dir, outname, PHASE=7, diaphragm_fraction=0.15):
    """
    Estimate diaphragm (lower-lung) downward motion - same computation as
    estimate_diaphragm_motion.ipynb. Requires the Register step to have been
    run already; raises FileNotFoundError (surfaced by run() as a clear
    step failure) if no registration is found.

    Saves into the registration output dir (Results/Registered/R{PHASE}_to_R{fix_phase}):
      - diaphragm_descent.html / _left.html / _right.html (interactive sanity-check plots)
      - diaphragm_descent_rotation.gif (rotating-camera GIF of the whole-lung plot;
        currently skipped - see GENERATE_DIAPHRAGM_GIF)
      - diaphragm_2D_visualization_coronal.jpg / _sagittal.jpg / _axial.jpg
        (static coronal/sagittal/axial projections of the same plot; arrows
        cover the whole lung)
      - diaphragm_2D_visualization_panel.jpg (all three side by side, one
        shared colorbar fixed to the same 0-5mm range as the standalone files)
      - ..._coronal_diaphragm.jpg / _sagittal_diaphragm.jpg / _axial_diaphragm.jpg
        / _panel_diaphragm.jpg - same four images again, each with a white
        highlight added over just the bottom diaphragm_fraction (where the
        reported descent number comes from)
      - diaphragm_analysis.txt (the descent numbers)
    """
    descent = get_diaphragm_descent_mm(main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction)
    descent_lr = get_diaphragm_descent_mm_lr(main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction)

    result = visualize_diaphragm_descent(main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction)
    result_left = visualize_diaphragm_descent(main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction, side="left")
    result_right = visualize_diaphragm_descent(main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction, side="right")

    out_dir = result["out_dir"]
    gif_out = os.path.join(out_dir, "diaphragm_descent_rotation.gif")
    if GENERATE_DIAPHRAGM_GIF:
        _save_diaphragm_rotation_gif(result["figure"], gif_out)
        print("rotating GIF saved to:", gif_out)
        log_artifact(gif_out)
    else:
        print("rotating GIF generation is paused (GENERATE_DIAPHRAGM_GIF=False) - skipping")

    # Each of the 4 image types (3 planes + panel) is saved twice: once
    # with arrows over the whole lung only (the default), and once with the
    # bottom-diaphragm_fraction white highlight also turned on - see
    # show_diaphragm_region on visualize_diaphragm_descent_2d /
    # _2d_panel. jpg_outs/jpg_outs_diaphragm are keyed the same way so the
    # txt summary below can list both cleanly.
    jpg_outs = {}
    jpg_outs_diaphragm = {}
    for plane in ("coronal", "sagittal", "axial"):
        jpg_outs[plane] = visualize_diaphragm_descent_2d(
            main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction, plane=plane,
            descent_left=descent_lr["left"], descent_right=descent_lr["right"])
        log_artifact(jpg_outs[plane])
        jpg_outs_diaphragm[plane] = visualize_diaphragm_descent_2d(
            main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction, plane=plane,
            descent_left=descent_lr["left"], descent_right=descent_lr["right"], show_diaphragm_region=True)
        log_artifact(jpg_outs_diaphragm[plane])
    jpg_out = jpg_outs["coronal"]  # kept as the top-level `jpg_out` return key

    panel_out = visualize_diaphragm_descent_2d_panel(
        main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction,
        descent_left=descent_lr["left"], descent_right=descent_lr["right"])
    log_artifact(panel_out)
    panel_out_diaphragm = visualize_diaphragm_descent_2d_panel(
        main_dir, outname, PHASE=PHASE, diaphragm_fraction=diaphragm_fraction,
        descent_left=descent_lr["left"], descent_right=descent_lr["right"], show_diaphragm_region=True)
    log_artifact(panel_out_diaphragm)

    ratio = (descent_lr["left"] / descent_lr["right"]) if descent_lr["right"] else float("nan")

    lines = [
        "Diaphragm motion analysis",
        f"main_dir: {main_dir}",
        f"outname: {outname}",
        f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"PHASE (moving): {PHASE}",
        f"fix_phase (fixed): {result['fix_phase']}",
        f"diaphragm_fraction: {diaphragm_fraction}",
        "",
        f"Average diaphragm descent (both lungs): {descent:.3f} mm",
        f"Left diaphragm descent:                 {descent_lr['left']:.3f} mm",
        f"Right diaphragm descent:                {descent_lr['right']:.3f} mm",
        f"Ratio left/right descent:                {ratio:.3f}",
        "",
        f"n_voxels (both):  {result['n_voxels']}",
        f"n_voxels (left):  {result_left['n_voxels']}",
        f"n_voxels (right): {result_right['n_voxels']}",
        "",
        f"HTML (both):  {result['html_out']}",
        f"HTML (left):  {result_left['html_out']}",
        f"HTML (right): {result_right['html_out']}",
        f"GIF: {gif_out if GENERATE_DIAPHRAGM_GIF else '(skipped - GENERATE_DIAPHRAGM_GIF is paused)'}",
        f"2D visualization (coronal projection): {jpg_outs['coronal']}",
        f"2D visualization (sagittal projection): {jpg_outs['sagittal']}",
        f"2D visualization (axial projection):    {jpg_outs['axial']}",
        f"2D visualization (panel, coronal+sagittal+axial): {panel_out}",
        f"2D visualization + diaphragm-region highlight (coronal): {jpg_outs_diaphragm['coronal']}",
        f"2D visualization + diaphragm-region highlight (sagittal): {jpg_outs_diaphragm['sagittal']}",
        f"2D visualization + diaphragm-region highlight (axial):    {jpg_outs_diaphragm['axial']}",
        f"2D visualization + diaphragm-region highlight (panel):    {panel_out_diaphragm}",
        "",
    ]
    txt_out = os.path.join(out_dir, "diaphragm_analysis.txt")
    with open(txt_out, "w") as f:
        f.write("\n".join(lines))
    print("analysis summary saved to:", txt_out)
    log_artifact(txt_out)

    return dict(descent=descent, descent_lr=descent_lr, out_dir=out_dir, gif_out=gif_out,
                jpg_out=jpg_out, jpg_outs=jpg_outs, jpg_outs_diaphragm=jpg_outs_diaphragm,
                panel_out=panel_out, panel_out_diaphragm=panel_out_diaphragm, txt_out=txt_out)


outname = 'Optimized_Reconstruction-MI'
# --------------------------------------------------------------------------
# End of code copied from the notebook.
# --------------------------------------------------------------------------


def run(main_dir, steps, threshold=0.5, announce_done=True):
    """announce_done=False suppresses the trailing "===ALL_DONE===" marker -
    used by run_group(), which calls this once per session, so the GUI
    doesn't mistake one session finishing for the whole group being done
    (run_group prints its own "===GROUP_ALL_DONE===" once every session in
    the group has actually run)."""
    if "recon" in steps:
        log_step("recon")
        try:
            MI_reconstruction(main_dir, threshold=threshold)
        except Exception as e:
            log_failed("recon", e)
            return 1
        log_done("recon")

    if "segment" in steps:
        log_step("segment")
        try:
            compress_files(main_dir, outname)
            segment(main_dir, outname)
            make_gif_cropped(main_dir, outname=outname, vmin=-1150, vmax=350, frame=-1)
            log_artifact(os.path.join(main_dir, outname, 'Results', 'images_cropped', 'video.gif'))
        except Exception as e:
            log_failed("segment", e)
            return 1
        log_done("segment")

    if "analysis" in steps:
        log_step("analysis")
        try:
            get_maps(main_dir, outname=outname, overwrite=False)
            log_artifact(os.path.join(main_dir, outname, 'Results', 'Maps', 'maps.png'))
        except Exception as e:
            log_failed("analysis", e)
            return 1
        log_done("analysis")

    if "register" in steps:
        log_step("register")
        try:
            register_step(main_dir, outname, EE_toEI_only=True)
            get_difformation_fields(main_dir, outname)
        except Exception as e:
            log_failed("register", e)
            return 1
        log_done("register")

    if "diaphragm" in steps:
        log_step("diaphragm")
        try:
            diaphragm_step(main_dir, outname)
        except Exception as e:
            log_failed("diaphragm", e)
            return 1
        log_done("diaphragm")

    if announce_done:
        print("\n===ALL_DONE===", flush=True)
    return 0


def get_rat_session_dirs(rat_dir):
    """
    Every session (date/time) folder under a rat's raw data directory, e.g.
    D:\\Data\\PhNd7 -> [D:\\Data\\PhNd7\\2026-07-20_11h01, ...], sorted
    chronologically. Same folder-naming pattern (and same discovery logic)
    as Desktop/Pipeline/Rat_all_dates_analysis.ipynb's get_dates(): matches
    "YYYY-MM-DD_HHhMM" directory names directly under the rat folder,
    regardless of whether reconstruction has been run there yet - unlike
    get_dates(), this doesn't need to know the outname, since this app
    always uses the single fixed `outname` above for every session.
    """
    import re
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}h\d{2}$")
    if not os.path.isdir(rat_dir):
        return []
    return sorted(
        os.path.join(rat_dir, d) for d in os.listdir(rat_dir)
        if os.path.isdir(os.path.join(rat_dir, d)) and date_pattern.match(d)
    )


def run_group(rat_dirs, steps, threshold=0.5):
    """
    Runs `steps` (the same set run() runs for one dataset) across every
    session found under each rat directory in `rat_dirs` - one rat at a
    time, one session at a time within each rat, in the order given. Same
    "loop over get_dates(rat)" pattern as
    Rat_all_dates_analysis.ipynb, just driving this script's existing
    per-session run() instead of the notebook's own inline cells.

    A session whose steps fail is logged (===SESSION_FAILED===) and the
    group moves on to the next session rather than aborting the whole
    group - a group run can cover many sessions across many rats, so one
    bad session shouldn't discard everything already completed. The
    overall return code is 1 if any session failed, 0 only if every
    session's every requested step succeeded.
    """
    sessions_by_rat = {rat_dir: get_rat_session_dirs(rat_dir) for rat_dir in rat_dirs}
    total_sessions = sum(len(s) for s in sessions_by_rat.values())
    print(f"\n===GROUP_START=== rats={len(rat_dirs)} sessions={total_sessions}", flush=True)

    overall_rc = 0
    done = 0
    for rat_dir in rat_dirs:
        rat_name = os.path.basename(os.path.normpath(rat_dir))
        sessions = sessions_by_rat[rat_dir]
        if not sessions:
            print(f"\n===RAT_SKIPPED=== {rat_name}: no session folders found under {rat_dir}", flush=True)
            continue
        for session_dir in sessions:
            done += 1
            print(f"\n===SESSION=== {session_dir} ({done}/{total_sessions})", flush=True)
            try:
                rc = run(session_dir, steps, threshold=threshold, announce_done=False)
            except Exception as e:
                print(f"===SESSION_FAILED=== {session_dir}: {e}", flush=True)
                traceback.print_exc()
                rc = 1
            if rc == 0:
                print(f"===SESSION_DONE=== {session_dir}", flush=True)
            else:
                overall_rc = 1
                print(f"===SESSION_FAILED=== {session_dir}: step(s) failed, see log above", flush=True)

    print(f"\n===GROUP_ALL_DONE=== {done}/{total_sessions} sessions processed", flush=True)
    return overall_rc


def main():
    parser = argparse.ArgumentParser(description="Run the FMIG rat reconstruction pipeline headlessly.")
    parser.add_argument("main_dir", nargs="?", default=None,
                         help=r'Data folder for single-dataset mode, e.g. D:\Data\Txk5\2026-08-06_10h47. '
                              r'Omit and use --rats instead for group mode.')
    parser.add_argument("--rats", action="append", default=None,
                         help=r'Rat directory for group mode, e.g. D:\Data\PhNd7 (contains that rat\'s '
                              r'session subfolders). Repeat --rats once per rat to run several. Every '
                              r'session found under each rat directory is run in turn (see '
                              r'get_rat_session_dirs). Mutually exclusive with the positional main_dir.')
    parser.add_argument("--steps", default=",".join(STEP_ORDER),
                         help="Comma-separated subset of: " + ",".join(STEP_ORDER))
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Projection segmentation threshold in (0, 1] used when "
                              "extracting the breathing signal for gated recon. Default 0.5.")
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEP_ORDER)
    if unknown:
        print(f"Unknown step(s): {unknown}. Valid steps: {STEP_ORDER}", file=sys.stderr)
        return 2

    threshold = args.threshold
    if not (0.0 < threshold <= 1.0):
        clamped = min(1.0, max(0.01, threshold))
        print(f"threshold {threshold} out of range (0, 1] - using {clamped}", file=sys.stderr)
        threshold = clamped

    if args.rats:
        if args.main_dir:
            print("main_dir and --rats are mutually exclusive - pass one or the other.", file=sys.stderr)
            return 2
        missing = [r for r in args.rats if not os.path.isdir(r)]
        if missing:
            print(f"Rat director{'y' if len(missing) == 1 else 'ies'} not found: {missing}", file=sys.stderr)
            return 2
        print(f"rats = {args.rats}")
        print(f"steps = {steps}")
        print(f"threshold = {threshold}")
        return run_group(args.rats, steps, threshold=threshold)

    if not args.main_dir:
        print("Either main_dir or --rats is required.", file=sys.stderr)
        return 2
    main_dir = args.main_dir
    if not os.path.isdir(main_dir):
        print(f"main_dir does not exist: {main_dir}", file=sys.stderr)
        return 2

    print(f"main_dir = {main_dir}")
    print(f"steps = {steps}")
    print(f"threshold = {threshold}")
    return run(main_dir, steps, threshold=threshold)


if __name__ == "__main__":
    sys.exit(main())
