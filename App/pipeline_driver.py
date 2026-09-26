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

Must be run with the "general app" environment's python.exe - the one
setup_environment.bat (repo root) builds at <install root>\python\python.exe
(fmig_rat_app.py resolves and passes this automatically; see its
resolve_general_python()). The notebook itself still uses its own "mi-env"
conda kernel (see kernel.json) - out of scope for setup_environment.bat.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

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

STEP_ORDER = ["recon", "segment", "segment_lr", "analysis", "register", "register_all_phases",
              "diaphragm", "volume_analysis", "register_to_baseline"]


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
from fmig_version import get_code_version, record_code_version
import subprocess as _subprocess  # keep builtin 'subprocess' import below too

import subprocess
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath


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


# Number of breathing (time) phases - shared by the clustering plot (step_4)
# and the reconstruction (create_milab_structure) so they bin identically.
TIME_PHASES = 16


def step_4(main_dir, xlimits, threshold):
    ts = time.time()
    s, t, a = get_signal_from_image(os.path.join(main_dir, 'ct-data', 'corr'), subtract_baseline=1,
                                     spring=False, rabbit=False, xlimits=xlimits, threshold=threshold)
    print(f'Took {round((time.time() - ts) / 60, 2)} minutes.')
    # Same binning code + settings as the reconstruction (create_milab_structure
    # -> new_binning -> run_time_binning), so this plot shows exactly the
    # labels that become the Phase_<n> folders.
    out_png = os.path.join(main_dir, 'clustering_result.png')
    run_time_binning(s, t, a, TIME_PHASES, only_phase_binning=0, plot_path=out_png)
    log_artifact(out_png)
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
            
def MI_reconstruction(main_dir, threshold=0.5, use_fallback_timing=False):
    global outname
    corr_dir = os.path.join(main_dir, 'ct-data', 'corr')
    # outname is decided per-session, from whether THIS session's own
    # timing actually needed the fallback - not just from the flag being
    # allowed - so a batch run over several sessions doesn't mislabel the
    # ones that never needed it (see resolve_outname/run()'s docstring).
    fallback_used = check_timing(main_dir, use_fallback=use_fallback_timing)
    outname = FALLBACK_OUTNAME if fallback_used else BASE_OUTNAME
    step_2(main_dir)
    xlimits = step_3(main_dir, corr_dir, threshold=threshold)
    s, t, a = step_4(main_dir, xlimits, threshold)
    create_milab_structure(main_dir,
                            intensity_phases=10,
                            time_phases=TIME_PHASES,
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


# A runtime-only subset of RTKRecon (recon_client.py, recon_server.py,
# recon_server_common.py, milabs_rtk_recon.py, start_recon_server.bat) is
# vendored into this repo under vendor/RTKRecon. The full RTKRecon project
# (notebooks, etc.) still lives at Desktop/Pipeline/RTKRecon.
RTKRECON_DIR = os.path.join(PIPELINE_DIR, "vendor", "RTKRecon")


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
          "(start vendor/RTKRecon/start_recon_server.bat to skip the itk/rtk warm-up next time).",
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


from register_phases import register_all_phases, find_bash, run_registration, DEFAULT_ANTSPATH


def _pipeline_config_path(main_dir, outname):
    return os.path.join(main_dir, outname, "Results", "pipeline_config.json")


def load_pipeline_config(main_dir, outname):
    path = _pipeline_config_path(main_dir, outname)
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_pipeline_config(main_dir, outname, cfg):
    path = _pipeline_config_path(main_dir, outname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def get_ee_phase(main_dir, outname, force_recompute=False):
    """
    The detected end-expiration phase (smallest lung mask volume across the
    16-phase respiratory cycle) for this dataset - cached in a per-dataset
    Results/pipeline_config.json so the Register step and every downstream
    analysis (FRC/TLC/TV/FV/J, diaphragm motion) agree on which phase is
    "EE", instead of each one separately assuming phase 7. That assumption
    doesn't always hold: confirmed on real data (PhNd31) that one session's
    true minimum-volume phase was 11, not 7 - only ~1% smaller in that
    particular case, but nothing guarantees it always will be, so this
    detects it per-dataset rather than hardcoding it.

    Requires mask.npy (i.e. the Segment step) to exist the first time this
    is called for a dataset; after that the result is read from the config
    file, no mask.npy access needed. Pass force_recompute=True to ignore a
    cached value and redetect (e.g. after re-running Segment).
    """
    cfg = load_pipeline_config(main_dir, outname)
    if not force_recompute and "ee_phase" in cfg:
        return cfg["ee_phase"]

    mask_path = os.path.join(main_dir, outname, "Results", "mask.npy")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(
            f"{mask_path} not found - run the Segment step first so the EE phase can be detected.")
    masks = np.load(mask_path)
    sizes = masks.sum(axis=(1, 2, 3))
    ee_phase = int(np.argmin(sizes))
    ei_phase = int(np.argmax(sizes))

    cfg["ee_phase"] = ee_phase
    cfg["ei_phase"] = ei_phase
    cfg["ee_phase_voxels"] = int(sizes[ee_phase])
    cfg["ei_phase_voxels"] = int(sizes[ei_phase])
    cfg["phase_voxel_counts"] = [int(s) for s in sizes]
    save_pipeline_config(main_dir, outname, cfg)
    print(f"Detected EE phase R{ee_phase} ({sizes[ee_phase]:,} voxels), "
          f"EI phase R{ei_phase} ({sizes[ei_phase]:,} voxels) - saved to {_pipeline_config_path(main_dir, outname)}")
    return ee_phase


def register_step(main_dir, outname, EE_toEI_only=False, phase=7):
    """
    phase defaults back to 7 (this pipeline's original convention), not the
    auto-detected EE phase from get_ee_phase() - pass phase=None to use
    that detection instead. See get_ee_phase's docstring: on at least one
    real dataset the true minimum-volume phase differed from 7, so this
    default is a deliberate simplicity/consistency choice, not a claim
    that phase 7 is always correct.
    """
    results_dir = os.path.join(main_dir, outname, 'Results')
    registered_dir = os.path.join(results_dir, 'Registered')
    os.makedirs(registered_dir, exist_ok=True)

    if EE_toEI_only:
        target_phase = phase if phase is not None else get_ee_phase(main_dir, outname)
        only_phases = [target_phase]
    else:
        only_phases = None
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


def _locate_analysis_maps_files(main_dir, outname, PHASE=7):
    """
    Find the Jacobian/Warped files from this pipeline's actual R{PHASE}_to_
    R{fix_phase} ANTs registration (see register_step/register_phases.py),
    plus R{fix_phase}'s own raw image and mask - everything
    save_maps_registrations_core's TV/FV/J math needs. Analysis.get_maps()
    expects a Results/Registration/registered.npy + jacobian.npy pair
    (one warped/Jacobian volume per phase, all resampled onto a common
    reference grid) that nothing in this pipeline ever produces, so its
    TV/FV/J step silently stayed a no-op even with Register done - this
    pulls the same information from the per-pair NIfTI files this pipeline
    actually writes instead. yi2.sh always resamples onto the *fixed*
    phase's own grid (R{fix_phase}, R0 in practice), so R{fix_phase}'s raw
    image/mask can be used directly with no warp needed - it's the "ei"
    (larger-volume) side; R{PHASE}'s Warped/Jacobian files are the "ee"
    (smaller-volume) side. See _locate_diaphragm_warp_and_mask for the
    matching Forward.nii.gz lookup used by the diaphragm-motion step.
    """
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

    # yi2.sh names its outputs "{fixname}_To_{movname}_<suffix>" where
    # fixname is R{PHASE} (the one that gets warped) and movname is
    # R{fix_phase} (the reference grid) - see yi2.sh's own "fixname is the
    # one will be warped" comment. A folder can also contain a stray
    # R{fix_phase}_To_R{PHASE}_Warped.nii.gz from an earlier run in the
    # opposite direction (confirmed on real data), so the direction tag
    # must be checked explicitly rather than matching on "_Warped.nii"/
    # "_Jacobian.nii" alone - a same-named file for the wrong direction
    # silently reads real CT values off the wrong grid instead of failing,
    # producing a plausible-looking but wrong TV/FV/J map.
    direction_tag = f"R{PHASE}_To_"
    jac_candidates = [os.path.join(out_dir, x) for x in os.listdir(out_dir)
                       if direction_tag in x and "_Jacobian.nii" in x]
    if not jac_candidates:
        raise FileNotFoundError(f"No '*{direction_tag}*_Jacobian.nii*' file found in {out_dir} - run the Register step first.")

    warped_candidates = [os.path.join(out_dir, x) for x in os.listdir(out_dir)
                          if direction_tag in x and (x.endswith("_Warped.nii.gz") or x.endswith("_Warped.nii"))]
    if not warped_candidates:
        raise FileNotFoundError(f"No '*{direction_tag}*_Warped.nii*' file found in {out_dir} - run the Register step first.")

    # TotalWarp.nii.gz (not the "_Forward" variant, hence the exact suffix
    # match) is the displacement field defined on R{fix_phase}'s (EI's) own
    # grid, pointing EI->EE - the EI-grid analogue of the Forward.nii.gz
    # field _locate_diaphragm_warp_and_mask uses on R{PHASE}'s (EE's) grid.
    # Used for the maps panels' whole-lung "Deformation" arrows.
    total_warp_candidates = [os.path.join(out_dir, x) for x in os.listdir(out_dir)
                              if direction_tag in x and x.endswith("_TotalWarp.nii.gz")]
    if not total_warp_candidates:
        raise FileNotFoundError(f"No '*{direction_tag}*_TotalWarp.nii.gz' file found in {out_dir} - run the Register step first.")

    fixed_mask_candidates = [os.path.join(RESULTS_DIR, x) for x in os.listdir(RESULTS_DIR)
                              if f"R{fix_phase}_m.nii" in x]
    if not fixed_mask_candidates:
        raise FileNotFoundError(f"No mask file matching 'R{fix_phase}_m.nii*' found in {RESULTS_DIR}")

    fixed_image_candidates = [os.path.join(RESULTS_DIR, x) for x in os.listdir(RESULTS_DIR)
                               if f"R{fix_phase}.nii" in x and "_m" not in x]
    if not fixed_image_candidates:
        raise FileNotFoundError(f"No raw image file matching 'R{fix_phase}.nii*' found in {RESULTS_DIR}")

    return dict(
        jacobian_path=jac_candidates[0],
        warped_path=warped_candidates[0],
        total_warp_path=total_warp_candidates[0],
        fixed_mask_path=fixed_mask_candidates[0],
        fixed_image_path=fixed_image_candidates[0],
        fix_phase=fix_phase,
    )


def _locate_ee_grid_transform_files(main_dir, outname, PHASE=7):
    """
    Everything needed to resample R{fix_phase} (EI, R0 in practice) onto
    R{PHASE}'s (EE's, R7's) own grid, plus the forward warp field needed to
    build a Jacobian on that same grid - the reverse of what
    _locate_analysis_maps_files gives (which puts everything on EI's grid,
    the direction yi2.sh's own Warped.nii.gz/Jacobian.nii.gz use). Reuses
    the *forward* transform chain (Warp2, Warp1, Affine) yi2.sh already
    computed and used to build its own TotalWarp_Forward.nii.gz (a field
    defined on R{PHASE}'s grid, pointing R{PHASE}->R{fix_phase}) - handing
    that same chain to antsApplyTransforms with an image instead of asking
    it to emit a field resamples that image onto R{PHASE}'s grid, with no
    new antsRegistration run needed.
    """
    base = _locate_analysis_maps_files(main_dir, outname, PHASE=PHASE)
    out_dir = os.path.dirname(base["jacobian_path"])
    RESULTS_DIR = os.path.join(main_dir, outname, "Results")
    direction_tag = f"R{PHASE}_To_"

    def _find_in_out_dir(suffix):
        cands = [os.path.join(out_dir, x) for x in os.listdir(out_dir)
                  if direction_tag in x and x.endswith(suffix)]
        if not cands:
            raise FileNotFoundError(f"No '*{direction_tag}*{suffix}' file found in {out_dir} - run the Register step first.")
        return cands[0]

    ee_reference_candidates = [os.path.join(RESULTS_DIR, x) for x in os.listdir(RESULTS_DIR)
                                if f"R{PHASE}.nii" in x and "_m" not in x]
    if not ee_reference_candidates:
        raise FileNotFoundError(f"No raw image file matching 'R{PHASE}.nii*' found in {RESULTS_DIR}")

    ee_mask_candidates = [os.path.join(RESULTS_DIR, x) for x in os.listdir(RESULTS_DIR)
                           if f"R{PHASE}_m.nii" in x]
    if not ee_mask_candidates:
        raise FileNotFoundError(f"No mask file matching 'R{PHASE}_m.nii*' found in {RESULTS_DIR}")

    return dict(
        warp2=_find_in_out_dir("outputPrefixResults2Warp.nii.gz"),
        warp1=_find_in_out_dir("outputPrefixResults1Warp.nii.gz"),
        affine=_find_in_out_dir("outputPrefixResults0GenericAffine.mat"),
        forward_field=_find_in_out_dir("TotalWarp_Forward.nii.gz"),
        ei_image_path=base["fixed_image_path"],       # R{fix_phase} (EI) raw image - to be resampled
        ee_reference_path=ee_reference_candidates[0],  # R{PHASE} (EE) raw image - defines the target grid
        ee_mask_path=ee_mask_candidates[0],
        out_dir=out_dir,
    )


def get_ei_on_ee_grid(main_dir, outname, PHASE=7, overwrite=False, antspath=None):
    """
    Resample R{fix_phase} (EI)'s raw image onto R{PHASE}'s (EE's) own grid,
    and compute a Jacobian determinant map natively on that same grid (the
    native EE->EI direction, >1) via antsApplyTransforms /
    CreateJacobianDeterminantImage directly (no new antsRegistration run -
    see _locate_ee_grid_transform_files). Cached next to the rest of the
    registration output; each file is skipped if it already exists and
    overwrite is False.
    """
    files = _locate_ee_grid_transform_files(main_dir, outname, PHASE=PHASE)
    out_dir = files["out_dir"]
    antspath = Path(antspath) if antspath else DEFAULT_ANTSPATH

    ei_on_ee_path = os.path.join(out_dir, "EI_on_EE_grid.nii.gz")
    jac_on_ee_path = os.path.join(out_dir, "EE_grid_Jacobian.nii.gz")

    if overwrite or not os.path.isfile(ei_on_ee_path):
        cmd = [
            str(antspath / "antsApplyTransforms.exe"), "-d", "3",
            "-r", files["ee_reference_path"],
            "-i", files["ei_image_path"],
            "-o", ei_on_ee_path,
            "-n", "Linear",
            "-t", files["warp2"],
            "-t", files["warp1"],
            "-t", files["affine"],
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"antsApplyTransforms failed (EI onto EE grid):\n{result.stdout}\n{result.stderr}")

    if overwrite or not os.path.isfile(jac_on_ee_path):
        cmd = [
            str(antspath / "CreateJacobianDeterminantImage.exe"), "3",
            files["forward_field"], jac_on_ee_path, "0", "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"CreateJacobianDeterminantImage failed (EE grid):\n{result.stdout}\n{result.stderr}")

    return dict(
        ei_on_ee_path=ei_on_ee_path,
        jac_on_ee_path=jac_on_ee_path,
        ee_reference_path=files["ee_reference_path"],
        ee_mask_path=files["ee_mask_path"],
    )


def _save_analysis_maps_from_pair(ei_img, ee_img, jac, mask, maps_dir, num_slices=6, dpi=300, affine=None):
    """
    Same TV/FV/J math as Analysis.save_maps_registrations_core, just fed a
    single already-registered (ei, ee, jacobian) triple instead of indexing
    into a 16-phase stack - see _locate_analysis_maps_files for where these
    come from in this pipeline. Also saves DeltaBlood.png/_smoothed.png (see
    get_maps_single_slice_panel_ei's DeltaBlood_3d for the same formula) when
    an affine is supplied to convert voxel volume to nL.
    """
    TV_path = os.path.join(maps_dir, 'TV.png')
    FV_path = os.path.join(maps_dir, 'FV.png')
    J_path = os.path.join(maps_dir, 'J.png')
    DeltaBlood_path = os.path.join(maps_dir, 'delta_blood.png')

    fx = mask.any(axis=(1, 2))
    fy = mask.any(axis=(0, 2))
    fz = mask.any(axis=(0, 1))
    pad = 4

    def _crop(img):
        out = (img * mask)[fx, :, :][:, fy, :][:, :, fz]
        out = np.pad(out, pad, 'constant', constant_values=0).astype(np.float32)
        out[out == 0] = np.nan
        return out

    ee_masked = _crop(ee_img)
    ei_masked = _crop(ei_img)
    J_masked = _crop(jac)

    voxel_vol_nL = None
    if affine is not None:
        zooms = np.abs(np.diag(affine))[:3]
        voxel_vol_nL = float(zooms[0] * zooms[1] * zooms[2]) * 1000.0  # mm^3 -> nL

    for smooth in (False, True):
        ee_slices = get_x_slices_from_image(ee_masked, number_of_slices=num_slices,
                                             coronal_axis=1, transpose=False, airways=False, smooth=smooth)
        ei_slices = get_x_slices_from_image(ei_masked, number_of_slices=num_slices,
                                             coronal_axis=1, transpose=False, airways=False, smooth=smooth)
        J_slices = get_x_slices_from_image(J_masked, number_of_slices=num_slices,
                                            coronal_axis=1, transpose=False, airways=False, smooth=smooth)
        FV = 1 - np.divide(ee_slices * J_slices, ei_slices, where=ei_slices != 0)
        J = 1 - J_slices
        # Jacobian-corrected TV (CT_Ventilation_Equations.pdf Convention B,
        # EI-domain: TV = (j*HU_EE - HU_EI)/1000, j = J_slices here) - matches
        # the same j-correction FV already applies to its ee_slices term.
        TV = (J_slices * ee_slices - ei_slices) / 1000
        if smooth:
            save_image_MI_style(FV, 0, 1, FV_path.replace(".png", "_smoothed.png"), dpi)
            save_image_MI_style(J, 0, 1, J_path.replace(".png", "_smoothed.png"), dpi)
            save_image_MI_style(TV, -0.1, 0.6, TV_path.replace(".png", "_smoothed.png"), dpi)
        else:
            save_image_MI_style(FV, 0, 1, FV_path, dpi)
            save_image_MI_style(J, 0, 1, J_path, dpi)
            save_image_MI_style(TV, -0.1, 0.6, TV_path, dpi)

        if voxel_vol_nL is not None:
            # CT_Ventilation_Equations.pdf Convention B (EI-domain): Delta
            # denotes EI minus EE - (HU_EI+1000) - j*(HU_EE+1000).
            DeltaBlood = (voxel_vol_nL / 1000.0) * ((ei_slices + 1000) - J_slices * (ee_slices + 1000))
            out_path = DeltaBlood_path.replace(".png", "_smoothed.png") if smooth else DeltaBlood_path
            save_image_MI_style(DeltaBlood, -1.5, 1.5, out_path, dpi, cmap="coolwarm")


def get_maps_r7_r0(main_dir, outname, PHASE=7, overwrite=False):
    """
    Fill in the TV/FV/J maps (see _save_analysis_maps_from_pair) from this
    pipeline's actual R{PHASE}_to_R{fix_phase} registration output, then
    rebuild the combined maps.png. Analysis.get_maps() already produced
    FRC/TLC by this point (from raw.npy/mask.npy, no registration needed) -
    this only adds the three maps that need Register to have run.
    """
    RESULTS_DIR = os.path.join(main_dir, outname, "Results")
    maps_dir = os.path.join(RESULTS_DIR, "Maps")
    os.makedirs(maps_dir, exist_ok=True)
    TV_path = os.path.join(maps_dir, 'TV.png')
    FV_path = os.path.join(maps_dir, 'FV.png')
    J_path = os.path.join(maps_dir, 'J.png')
    DeltaBlood_path = os.path.join(maps_dir, 'delta_blood.png')
    if not overwrite and all(os.path.isfile(p) for p in (TV_path, FV_path, J_path, DeltaBlood_path)):
        return

    files = _locate_analysis_maps_files(main_dir, outname, PHASE=PHASE)

    fixed_mask_nii = nib.load(files["fixed_mask_path"])
    ei_img = nib.load(files["fixed_image_path"]).get_fdata()
    ee_img = nib.load(files["warped_path"]).get_fdata()
    jac = nib.load(files["jacobian_path"]).get_fdata()
    mask = fixed_mask_nii.get_fdata().astype(bool)

    ei_img = gaussian_filter(ei_img, 1)
    ee_img = gaussian_filter(ee_img, 1)
    jac = gaussian_filter(jac, 1)

    _save_analysis_maps_from_pair(ei_img, ee_img, jac, mask, maps_dir, affine=fixed_mask_nii.affine)
    save_combined_maps_figure(maps_dir)


def _mask_bbox_crop(img, mask, pad=4):
    # Same bounding-box crop as _save_analysis_maps_from_pair's local
    # _crop() (mask, crop to the mask's own bounding box, NaN the
    # background) - factored out here so get_maps_single_slice_panel can
    # crop ei/ee/jacobian once and compute TV/FV/J directly on the still-3D
    # result, instead of on an already-2D projection.
    fx = mask.any(axis=(1, 2))
    fy = mask.any(axis=(0, 2))
    fz = mask.any(axis=(0, 1))
    out = (img * mask)[fx, :, :][:, fy, :][:, :, fz]
    out = np.pad(out, pad, 'constant', constant_values=0).astype(np.float32)
    out[out == 0] = np.nan
    return out


# Which voxel axis each plane collapses to make its projection - matches
# the "coronal_axis" convention already used throughout this pipeline
# (Analysis.py/utils.py's get_x_slices_from_image calls all pass
# coronal_axis=1 for this same data), extended the same way to sagittal
# (collapse left/right) and axial (collapse superior/inferior).
MAPS_PLANE_AXIS = {"coronal": 1, "sagittal": 0, "axial": 2}


def _masked_axis_projection(image_3d, mask_3d, axis=1):
    # Mean across the whole named axis instead of picking out individual
    # slices, so the single 2D result reflects the entire lung volume, not
    # just one cross-section through it - same idea as the diaphragm step's
    # coronal *projection* (Y collapsed), just applied to a functional map
    # instead of a point cloud, and generalized to any of the 3 planes.
    # utils.get_x_slices_from_image has an "airways" mode meant to do
    # exactly this, but it calls a filter_nan_gaussian_conserving() that
    # doesn't exist anywhere in this codebase (pre-existing dead code,
    # unrelated to this feature) - np.nanmean directly is the same core
    # operation without that broken optional smoothing step.
    masked = image_3d * mask_3d
    cropped = crop_to_mask(masked, padding=4)
    cropped[cropped == 0] = np.nan
    with np.errstate(invalid="ignore"):
        return np.nanmean(cropped, axis=axis)


def _coronal_projection(image_3d, mask_3d):
    return _masked_axis_projection(image_3d, mask_3d, axis=MAPS_PLANE_AXIS["coronal"])


# Which two world axes (0=X/L-R, 1=Y/A-P, 2=Z/S-I) are horizontal/vertical
# for each plane's arrow projection - same convention as the diaphragm
# step's PLANE_CONFIG (h/v), just keyed simply for this narrower use.
DEFORMATION_PLANE_HV = {"coronal": (0, 2), "sagittal": (1, 2), "axial": (0, 1)}


def _block_average_arrows(warp, affine, ai, aj, ak, ARROW_BLOCK=15, MIN_VOXELS_PER_ARROW=4):
    """
    Block-averages a displacement field over ARROW_BLOCK-voxel neighborhoods
    of the given voxel indices, returning one arrow per block: its world-
    space position, mean displacement vector, and full 3D magnitude (used
    for arrow color). Same block-averaging the diaphragm 2D visualizations'
    whole-lung arrows use (_diaphragm_2d_arrays) - duplicated here rather
    than shared, to avoid touching that already-tested function for an
    unrelated feature.
    """
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
    if keep_blocks.sum() == 0:
        keep_blocks = counts >= 1
    mean_vecs = vec_sums[keep_blocks] / counts[keep_blocks, None]
    mean_idx = idx_sums[keep_blocks] / counts[keep_blocks, None]
    arrow_vox = np.concatenate([mean_idx, np.ones((len(mean_idx), 1))], axis=1)
    arrow_world = (affine @ arrow_vox.T).T[:, :3]
    arrow_norm = np.linalg.norm(mean_vecs, axis=1)
    return arrow_world, mean_vecs, arrow_norm


def _render_deformation_panel(ax, warp, mask, affine, plane, vmin=0.0, vmax=5.0):
    """
    Draws whole-lung block-averaged displacement arrows (colored by full 3D
    magnitude) over a gray lung silhouette, projected onto `plane` - the
    maps panels' "Deformation" column. Same visual idea as the diaphragm
    step's whole-lung arrows (diaphragm_2D_visualization_*.jpg), reusing
    _block_average_arrows instead of _diaphragm_2d_arrays since this needs
    no side-split, diaphragm-region highlighting, or orientation labels -
    just the arrows and enough context to see where they sit in the lung.
    """
    h_idx, v_idx = DEFORMATION_PLANE_HV[plane]

    ai, aj, ak = np.where(mask)
    stride_n = max(2 ** 3, 1)
    si, sj, sk = ai[::stride_n], aj[::stride_n], ak[::stride_n]
    vox = np.stack([si, sj, sk, np.ones_like(si)], axis=1).astype(float)
    world = (affine @ vox.T).T[:, :3]

    arrow_world, mean_vecs, arrow_norm = _block_average_arrows(warp, affine, ai, aj, ak)

    ax.set_facecolor("black")
    ax.scatter(world[:, h_idx], world[:, v_idx], s=4, c="gray", alpha=0.3)
    q = ax.quiver(arrow_world[:, h_idx], arrow_world[:, v_idx],
                  mean_vecs[:, h_idx], mean_vecs[:, v_idx], arrow_norm,
                  cmap="jet", angles="xy", scale_units="xy", scale=1,
                  width=0.012, headwidth=4.5, headlength=5.5, headaxislength=5, zorder=5)
    q.set_clim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return q


# One colormap per functional map, so each is visually distinct at a
# glance instead of every quantity sharing the same jet scale.
MAP_COLORMAPS = {
    "HU": "gray",  # raw CT signal, not a derived functional quantity - grayscale like any CT viewer
    "FRC": "rainbow",
    "TLC": "plasma",
    "TV": "viridis",
    "FV": "hot",
    "J": "magma",
}


def _render_maps_panel(panels, out_path, title, plane="coronal"):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), dpi=200, facecolor="black")
    axes = np.atleast_1d(axes)
    for ax, (name, arr, vmin, vmax) in zip(axes, panels):
        if name == "Deformation":
            warp, mask, affine = arr
            im = _render_deformation_panel(ax, warp, mask, affine, plane, vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(arr.T, origin="lower", cmap=MAP_COLORMAPS.get(name, "jet"), vmin=vmin, vmax=vmax)
            ax.axis("off")
        ax.set_title(name, color="white", fontsize=18)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white")
    fig.suptitle(title, color="white", fontsize=16)
    plt.savefig(out_path, bbox_inches="tight", dpi=200, facecolor="black")
    plt.close(fig)


def get_maps_single_slice_panel_ei(main_dir, outname, PHASE=7, plane="coronal", overwrite=False):
    """
    One combined image, single `plane` projection per map (FRC, TLC, TV, FV,
    J), everything expressed on R{fix_phase}'s (EI's, R0 in practice) own
    grid - the direction yi2.sh's own Warped.nii.gz/Jacobian.nii.gz come in,
    so no resampling needed here (see get_maps_single_slice_panel_ee for the
    EE-anchored counterpart, which does need an extra resampling step). Uses
    the native EI->EE Jacobian (<1) exactly as get_maps_r7_r0/maps.png do.
    """
    if plane not in MAPS_PLANE_AXIS:
        raise ValueError(f"plane must be one of {sorted(MAPS_PLANE_AXIS)}, got {plane!r}")
    axis = MAPS_PLANE_AXIS[plane]

    RESULTS_DIR = os.path.join(main_dir, outname, "Results")
    maps_dir = os.path.join(RESULTS_DIR, "Maps")
    os.makedirs(maps_dir, exist_ok=True)
    out_path = os.path.join(maps_dir, f"maps_panel_EI_{plane}.jpg")
    if not overwrite and os.path.isfile(out_path):
        return out_path

    files = _locate_analysis_maps_files(main_dir, outname, PHASE=PHASE)
    ei_mask_nii = nib.load(files["fixed_mask_path"])
    ei_mask = ei_mask_nii.get_fdata().astype(bool)
    ei_own_img = gaussian_filter(nib.load(files["fixed_image_path"]).get_fdata(), 1)
    ee_on_ei_img = gaussian_filter(nib.load(files["warped_path"]).get_fdata(), 1)
    jac_on_ei = gaussian_filter(nib.load(files["jacobian_path"]).get_fdata(), 1)  # native EI->EE, <1

    panels = [
        # Raw HU signal (this grid's own native image, unconverted) - just
        # the anatomical CT slice for reference, gray like any CT viewer.
        # Range calibrated to real percentiles on a real session (lung
        # tissue here runs about -990 to +120 HU at the 0.5-99.5th pctile).
        ("HU", _masked_axis_projection(ei_own_img, ei_mask, axis=axis), -1000, 200),
        # vmin/vmax below are calibrated to where this data actually lives
        # (checked via real percentiles on a real session), not just each
        # map's theoretical 0-1 bound - stretching the colormap over range
        # the data never reaches makes the whole panel look nearly flat.
        ("FRC", _masked_axis_projection(ee_on_ei_img / -1000, ei_mask, axis=axis), 0, 0.8),
        ("TLC", _masked_axis_projection(ei_own_img / -1000, ei_mask, axis=axis), 0, 0.9),
    ]

    # TV/FV/J computed voxel-wise in 3D first, THEN collapsed to 2D by
    # averaging - see get_maps_single_slice_panel_ee's longer comment on why
    # (FV's ratio is nonlinear, so the order matters for it specifically).
    ei_masked = _mask_bbox_crop(ei_own_img, ei_mask)
    ee_masked = _mask_bbox_crop(ee_on_ei_img, ei_mask)
    jac_masked = _mask_bbox_crop(jac_on_ei, ei_mask)

    with np.errstate(invalid="ignore", divide="ignore"):
        FV_3d = 1 - np.divide(ee_masked * jac_masked, ei_masked, where=ei_masked != 0)
    FV_3d = np.clip(FV_3d, 0.0, 1.0)  # see the EE panel's comment - a handful of near-zero-denominator voxels can otherwise wreck a whole averaged column
    J_map_3d = 1 - jac_masked
    # Jacobian-corrected TV (CT_Ventilation_Equations.pdf Convention B,
    # EI-domain: TV = (j*HU_EE - HU_EI)/1000, j = jac_masked here) - matches
    # the same j-correction FV_3d already applies to its ee_masked term.
    TV_3d = (jac_masked * ee_masked - ei_masked) / 1000

    with np.errstate(invalid="ignore"):
        # Range widened/re-centered after adding the Jacobian correction
        # (checked via real percentiles): TV can now legitimately dip
        # slightly negative (matches the document's note that regional TV
        # can be negative), and its upper tail runs further than the old
        # uncorrected 0-0.25.
        panels.append(("TV", np.nanmean(TV_3d, axis=axis), -0.1, 0.6))
        panels.append(("FV", np.nanmean(FV_3d, axis=axis), 0, 0.7))
        panels.append(("J", np.nanmean(J_map_3d, axis=axis), 0, 0.3))

    # Whole-lung displacement arrows, on EI's own grid: TotalWarp.nii.gz
    # points EI->EE (see _locate_analysis_maps_files), the EI-grid analogue
    # of the Forward.nii.gz field the EE panel uses.
    total_warp = np.asarray(nib.load(files["total_warp_path"]).dataobj).squeeze()
    panels.append(("Deformation", (total_warp, ei_mask, ei_mask_nii.affine), 0.0, 5.0))

    _render_maps_panel(panels, out_path, f"R0 (EI) functional maps - single {plane} projection", plane=plane)
    return out_path


def get_maps_single_slice_panel_ee(main_dir, outname, PHASE=7, plane="coronal", overwrite=False):
    """
    EE-anchored counterpart of get_maps_single_slice_panel_ei: every map
    expressed on R{PHASE}'s (EE's, R7 in practice) own grid instead, using
    EI resampled onto that grid and the *native* EE->EI Jacobian (>1,
    "how much did this bit of EE tissue expand to become EI") - see
    get_ei_on_ee_grid. Both FV and J use this same native Jacobian, not an
    inverted/resampled EI->EE one, so this panel is self-contained: every
    quantity in it is computed on EE's own grid using EE->EI-direction
    fields, mirroring how the EI panel uses only EI->EE-direction ones.

    FV keeps the EI panel's exact formula shape (1-(ee*jac_EI->EE)/ei) -
    dividing by the native EE->EI jacobian instead of multiplying by an
    EI->EE one is the same arithmetic (x/j == x*(1/j)), so this already
    uses jac_on_ee directly; there's no separate "inverted" copy of it.
    J displays jac_on_ee itself, uncapped and >1, since the user wants the
    expansion factor shown as what it actually is rather than remapped
    into the EI panel's 0-1 "amount of shrinkage" framing.
    """
    if plane not in MAPS_PLANE_AXIS:
        raise ValueError(f"plane must be one of {sorted(MAPS_PLANE_AXIS)}, got {plane!r}")
    axis = MAPS_PLANE_AXIS[plane]

    RESULTS_DIR = os.path.join(main_dir, outname, "Results")
    maps_dir = os.path.join(RESULTS_DIR, "Maps")
    os.makedirs(maps_dir, exist_ok=True)
    out_path = os.path.join(maps_dir, f"maps_panel_EE_{plane}.jpg")
    if not overwrite and os.path.isfile(out_path):
        return out_path

    raw_path = os.path.join(RESULTS_DIR, 'raw.npy')
    mask_path = os.path.join(RESULTS_DIR, 'mask.npy')
    if not os.path.isfile(raw_path) or not os.path.isfile(mask_path):
        raise FileNotFoundError(f"raw.npy/mask.npy not found in {RESULTS_DIR} - run the Segment step first.")

    imgs = np.load(raw_path)
    masks = np.load(mask_path)
    sizes = masks.sum(axis=(1, 2, 3))
    ee_index = int(np.argmin(sizes))

    # vmin/vmax below are calibrated to where this data actually lives
    # (checked via real percentiles on a real session), not just each
    # map's theoretical 0-1 bound - stretching the colormap over range the
    # data never reaches makes the whole panel look nearly flat.
    ee_native_img = gaussian_filter(imgs[ee_index].astype(np.float32), 1)
    panels = [
        # Raw HU signal (this grid's own native image, unconverted) - see
        # the EI panel's HU comment for the range calibration.
        ("HU", _masked_axis_projection(ee_native_img, masks[ee_index], axis=axis), -1000, 200),
        ("FRC", _masked_axis_projection(ee_native_img / -1000, masks[ee_index], axis=axis), 0, 0.8),
    ]

    ee_grid_files = get_ei_on_ee_grid(main_dir, outname, PHASE=PHASE, overwrite=False)
    ee_mask_nii = nib.load(ee_grid_files["ee_mask_path"])
    ee_mask = ee_mask_nii.get_fdata().astype(bool)
    ei_on_ee_img = gaussian_filter(nib.load(ee_grid_files["ei_on_ee_path"]).get_fdata(), 1)
    ee_own_img = gaussian_filter(nib.load(ee_grid_files["ee_reference_path"]).get_fdata(), 1)
    jac_on_ee = gaussian_filter(nib.load(ee_grid_files["jac_on_ee_path"]).get_fdata(), 1)  # native EE->EI, >1

    panels.append(("TLC", _masked_axis_projection(ei_on_ee_img / -1000, ee_mask, axis=axis), 0, 0.9))

    ei_masked = _mask_bbox_crop(ei_on_ee_img, ee_mask)
    ee_masked = _mask_bbox_crop(ee_own_img, ee_mask)
    jac_masked = _mask_bbox_crop(jac_on_ee, ee_mask)  # native EE->EI, >1 - used directly for both FV and J

    with np.errstate(invalid="ignore", divide="ignore"):
        FV_3d = 1 - np.divide(ee_masked, ei_masked * jac_masked, where=(ei_masked * jac_masked) != 0)
    # FV is a per-voxel ratio, so any voxel where the denominator lands very
    # close to zero (confirmed on real data - a cluster of voxels near the
    # trachea wall, where smoothing pulls the HU value right through zero)
    # blows up to +-1000s. That's a handful of voxels, but enough to drag
    # np.nanmean for their entire AP column once projected to 2D, showing up
    # as an isolated blown-out streak rather than a smooth map. Clamping to
    # FV's own displayed range (0-1) keeps that from propagating.
    FV_3d = np.clip(FV_3d, 0.0, 1.0)
    # J shows the native expansion jacobian directly, unmodified - it's
    # generally >1, so no clipping to a 0-1 range like the other maps.
    J_map_3d = jac_masked
    # Jacobian-corrected TV (CT_Ventilation_Equations.pdf Convention A,
    # EE-domain: TV = (HU_EE - J*HU_EI)/1000, J = jac_masked here) - matches
    # the same J-correction FV_3d already applies to its ei_masked term.
    TV_3d = (ee_masked - jac_masked * ei_masked) / 1000

    with np.errstate(invalid="ignore"):
        # Range widened/re-centered after adding the Jacobian correction -
        # see the EI panel's identical comment.
        panels.append(("TV", np.nanmean(TV_3d, axis=axis), -0.1, 0.6))
        panels.append(("FV", np.nanmean(FV_3d, axis=axis), 0, 0.7))
        # Fixed scale (not a per-session dynamic max) so J stays comparable
        # across sessions/rats, same as every other map here using a fixed
        # vmin/vmax - real data on this dataset ranged about 0.94-1.41, so
        # 1.0-1.45 uses most of the colorbar instead of the 1.0-2.0 range
        # this used to have, where real data never got past the bottom 41%.
        panels.append(("J", np.nanmean(J_map_3d, axis=axis), 1.0, 1.45))

    # Whole-lung displacement arrows, on EE's own grid: Forward.nii.gz
    # points EE->EI (see _locate_diaphragm_warp_and_mask), the same field
    # the diaphragm step's whole-lung arrows already use.
    forward_warp_path, _, _, _ = _locate_diaphragm_warp_and_mask(main_dir, outname, PHASE)
    forward_warp = np.asarray(nib.load(forward_warp_path).dataobj).squeeze()
    panels.append(("Deformation", (forward_warp, ee_mask, ee_mask_nii.affine), 0.0, 5.0))

    _render_maps_panel(panels, out_path, f"R{PHASE} (EE) functional maps - single {plane} projection", plane=plane)
    return out_path


def _stack_panel_images_vertically(image_paths, out_path):
    # Stacks already-rendered panel JPGs (each a full row of maps with its
    # own colorbars/title) directly, rather than re-plotting from scratch -
    # same pattern Analysis.save_combined_maps_figure uses to build maps.png
    # from FRC.png/TLC.png/etc.
    from PIL import Image as PILImage
    imgs = [PILImage.open(p).convert("RGB") for p in image_paths]
    max_w = max(im.width for im in imgs)
    padded = []
    for im in imgs:
        if im.width < max_w:
            canvas = PILImage.new("RGB", (max_w, im.height), (0, 0, 0))
            canvas.paste(im, ((max_w - im.width) // 2, 0))
            im = canvas
        padded.append(im)
    total_h = sum(im.height for im in padded)
    out = PILImage.new("RGB", (max_w, total_h), (0, 0, 0))
    y = 0
    for im in padded:
        out.paste(im, (0, y))
        y += im.height
    out.save(out_path)


def get_maps_all_slice_panels(main_dir, outname, PHASE=7, overwrite=False):
    """
    Generates every single-slice map panel this pipeline supports: coronal/
    sagittal/axial for both the EI and EE frames (6 files), plus one
    combined "*_all_slice.jpg" per frame stacking coronal (top), sagittal
    (middle), and axial (bottom) - 2 more files, 8 total. Returns a dict of
    all 8 paths, keyed "{EI,EE}_{coronal,sagittal,axial}" and
    "{EI,EE}_all_slice".
    """
    RESULTS_DIR = os.path.join(main_dir, outname, "Results")
    maps_dir = os.path.join(RESULTS_DIR, "Maps")

    paths = {}
    for frame, panel_fn in (("EI", get_maps_single_slice_panel_ei), ("EE", get_maps_single_slice_panel_ee)):
        plane_paths = []
        for plane in ("coronal", "sagittal", "axial"):
            p = panel_fn(main_dir, outname, PHASE=PHASE, plane=plane, overwrite=overwrite)
            paths[f"{frame}_{plane}"] = p
            plane_paths.append(p)

        all_slice_path = os.path.join(maps_dir, f"maps_panel_{frame}_all_slice.jpg")
        if overwrite or not os.path.isfile(all_slice_path):
            _stack_panel_images_vertically(plane_paths, all_slice_path)
        paths[f"{frame}_all_slice"] = all_slice_path

    return paths


def build_rat_coronal_maps_panel(rat_dir, outname, frame="EI"):
    """
    Stacks every session's own coronal functional-maps panel
    (maps_panel_{frame}_coronal.jpg from get_maps_single_slice_panel_ei/_ee
    - FRC, TLC, TV, FV, J, Deformation columns) into one combined image
    under <rat_dir>/Analysis, one row per session in chronological order
    (top = earliest/first timepoint, same ordering as
    get_rat_session_dirs) - lets you see how each functional map changes
    session to session at a glance, each row labeled with its session
    name. `frame` is "EI" (R0-anchored) or "EE" (R7-anchored, in
    practice) - the two panel flavors get_maps_single_slice_panel_ei/_ee
    produce; call this once for each to get both files. Rebuilt from
    scratch every time (like volume_analysis_step's CSV, just as a fresh
    render rather than an upsert, since row images can't be edited in
    place), so it always reflects whichever sessions currently have that
    panel. Sessions that haven't run Analysis yet (no
    maps_panel_{frame}_coronal.jpg) are skipped rather than failing the
    whole thing. Returns None if no session has that panel yet.
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont

    rows = []
    for session_dir in get_rat_session_dirs(rat_dir):
        panel_path = os.path.join(session_dir, outname, "Results", "Maps", f"maps_panel_{frame}_coronal.jpg")
        if os.path.isfile(panel_path):
            rows.append((os.path.basename(os.path.normpath(session_dir)), panel_path))
        else:
            print(f"Rat {frame} coronal maps panel: skipping {session_dir} "
                  f"(no maps_panel_{frame}_coronal.jpg - run Analysis there first).")
    if not rows:
        print(f"Rat {frame} coronal maps panel: no session panels found under {rat_dir} - nothing to combine.")
        return None

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except Exception:
        font = ImageFont.load_default()

    labeled = []
    for session_name, panel_path in rows:
        im = PILImage.open(panel_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        bbox = draw.textbbox((0, 0), session_name, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([0, 0, tw + 16, th + 16], fill=(0, 0, 0))
        draw.text((8, 8), session_name, fill=(255, 255, 0), font=font)
        labeled.append(im)

    max_w = max(im.width for im in labeled)
    padded = []
    for im in labeled:
        if im.width < max_w:
            canvas = PILImage.new("RGB", (max_w, im.height), (0, 0, 0))
            canvas.paste(im, ((max_w - im.width) // 2, 0))
            im = canvas
        padded.append(im)
    total_h = sum(im.height for im in padded)
    combined = PILImage.new("RGB", (max_w, total_h), (0, 0, 0))
    y = 0
    for im in padded:
        combined.paste(im, (0, y))
        y += im.height

    analysis_dir = os.path.join(rat_dir, "Analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    out_path = os.path.join(analysis_dir, f"coronal_maps_all_sessions_{frame}.jpg")
    combined.save(out_path)
    return out_path


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


def _higher_worldx_is_right(affine):
    # Same empirically-corrected laterality _split_diaphragm_left_right
    # uses (the header's own axis code reads backwards for this pipeline's
    # data), just re-expressed for world-X comparisons instead of raw
    # voxel-index comparisons, since the manual split/airway lines are
    # drawn and stored in world mm coordinates.
    axis0_code = nib.aff2axcodes(affine)[0]
    higher_index_is_right = (axis0_code != "R")
    return higher_index_is_right if affine[0, 0] > 0 else (not higher_index_is_right)


def _split_override_path(main_dir, outname):
    return os.path.join(main_dir, outname, "Results", "lr_split_override.json")


def load_lr_split_override(main_dir, outname):
    """
    A manually-drawn replacement for _split_diaphragm_left_right's single-
    threshold heuristic, saved by the app's Left/Right Split editor - see
    save_lr_split_override. Returns None if this session has no override
    (the normal case), in which case callers should fall back to the
    automatic method.
    """
    path = _split_override_path(main_dir, outname)
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_lr_split_override(main_dir, outname, split_points, airway_points,
                            split_slices=None, airway_slices=None):
    """
    split_points: list of [x_mm, y_mm] world-coordinate points in the
    axial (X/Y) plane, drawn by the user - interpolated as a piecewise-
    linear curve giving the L/R boundary's X position as a function of Y
    (anterior-posterior), so it can bend to follow the mediastinal gap's
    actual shape instead of the automatic method's single global X
    threshold. This is the coarse, whole-volume fallback.
    airway_points: list of [x_mm, z_mm] world-coordinate points in the
    coronal (X/Z) plane, forming a closed polygon around the trachea/main
    bronchi - at least 3 points to enclose an area. Any voxel whose (X,
    Z) position falls inside this polygon is excluded from both lungs
    instead of being arbitrarily assigned to whichever side of the split
    it happens to fall on. Also a coarse, whole-volume fallback.
    split_slices/airway_slices: optional dicts of {array Y-index (as a
    string): points} - per-coronal-slice refinements drawn one slice at a
    time (App.LRSplitEditor's "This slice only" scope), each entry a line
    (split, [x_mm, z_mm] points) or closed polygon (airway) in that one
    slice's own X/Z plane. When either is non-empty it takes over
    entirely from the matching whole-volume fallback above - see
    classify_lr_with_override for exactly how.
    """
    if len(split_points) < 2 and not split_slices:
        raise ValueError("split_points needs at least 2 points to define a line "
                          "(or at least one per-slice split line).")
    path = _split_override_path(main_dir, outname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(split_points=split_points, airway_points=airway_points,
                        split_slices=split_slices or {}, airway_slices=airway_slices or {}), f, indent=2)
    return path


def _interp_line_x(points, coord_values):
    # Piecewise-linear interpolation of a hand-drawn line's X position at
    # given values of its independent coordinate (Y for the split line,
    # Z for the airway line - see classify_lr_with_override) - np.interp
    # requires increasing independent values and clamps outside the drawn
    # range to the nearest endpoint rather than extrapolating, which is
    # the sensible behavior for a line the user only drew over part of
    # the lung's full extent.
    pts = sorted(points, key=lambda p: p[1])
    coords = np.array([p[1] for p in pts], dtype=float)
    xs = np.array([p[0] for p in pts], dtype=float)
    return np.interp(coord_values, coords, xs)


def _axis_for_world_row(affine, row):
    # Which array axis (0,1,2) a world coordinate row (0=X,1=Y,2=Z) comes
    # from - the column with the dominant entry in that affine row. Valid
    # for the axis-aligned (no rotation/oblique) affines this whole
    # override system already assumes (see _higher_worldx_is_right).
    return int(np.argmax(np.abs(affine[row, :3])))


def _world_to_slice_index(world_coord, affine, row, axis):
    return np.round((world_coord - affine[row, 3]) / affine[row, axis]).astype(int)


def _nearest_edited_slice(slice_idx, edited_keys_sorted):
    # For each voxel's own Y-slice index, which EDITED slice (from a
    # sparse, possibly tiny set) is closest - lets a correction on a
    # handful of slices propagate to every untouched slice around it,
    # instead of requiring the user to redraw every single slice by hand.
    if len(edited_keys_sorted) == 1:
        return np.full(slice_idx.shape, edited_keys_sorted[0])
    pos = np.searchsorted(edited_keys_sorted, slice_idx)
    pos = np.clip(pos, 1, len(edited_keys_sorted) - 1)
    left = edited_keys_sorted[pos - 1]
    right = edited_keys_sorted[pos]
    return np.where((right - slice_idx) < (slice_idx - left), right, left)


def classify_lr_with_override(world_x, world_y, world_z, override, affine):
    """
    world_x/world_y/world_z: 1D arrays of voxel world coordinates.
    override: dict from load_lr_split_override (or freshly constructed the
    same shape while editing). Two ways to define each boundary:

      - a single whole-volume curve/polygon (split_points: X as f(Y),
        drawn on the axial X/Y projection; airway_points: a closed X/Z
        polygon, drawn on the coronal projection) applied to every voxel
        regardless of where it sits along Y - the original, coarse
        method.
      - per-coronal-slice refinements (split_slices/airway_slices: dicts
        of {array Y-index (str): points}), each a line (split) or closed
        polygon (airway) drawn in that one slice's own X/Z plane (see
        App.LRSplitEditor's "This slice only" scope). When either dict is
        non-empty, EVERY voxel uses whichever edited slice is nearest to
        it along Y for that boundary - not just voxels that fall exactly
        on an edited slice - so touching up a handful of slices refines
        the neighborhood around them too, and this replaces the matching
        whole-volume fallback entirely rather than blending with it.

    Returns (is_left, is_right, is_airway) boolean arrays of the same
    length - airway voxels are excluded from both is_left and is_right,
    matching how a manually-traced airway should be handled (neither
    lung, not double-counted).
    """
    higher_worldx_is_right = _higher_worldx_is_right(affine)
    y_axis = _axis_for_world_row(affine, 1)

    split_slices = override.get("split_slices") or {}
    if split_slices:
        keys = np.array(sorted(int(k) for k in split_slices.keys()))
        slice_idx = _world_to_slice_index(world_y, affine, 1, y_axis)
        nearest = _nearest_edited_slice(slice_idx, keys)
        split_x = np.zeros(world_x.shape, dtype=float)
        for k in keys:
            m = nearest == k
            if m.any():
                split_x[m] = _interp_line_x(split_slices[str(int(k))], world_z[m])
    else:
        split_x = _interp_line_x(override["split_points"], world_y)
    is_right_side = (world_x > split_x) if higher_worldx_is_right else (world_x < split_x)

    is_airway = np.zeros(world_x.shape, dtype=bool)
    airway_slices = override.get("airway_slices") or {}
    if airway_slices:
        keys = np.array(sorted(int(k) for k in airway_slices.keys()))
        slice_idx = _world_to_slice_index(world_y, affine, 1, y_axis)
        nearest = _nearest_edited_slice(slice_idx, keys)
        for k in keys:
            pts = airway_slices[str(int(k))]
            if len(pts) < 3:
                continue
            m = nearest == k
            if m.any():
                polygon = MplPath(np.asarray(pts, dtype=float))
                is_airway[m] = polygon.contains_points(np.column_stack([world_x[m], world_z[m]]))
    else:
        airway_points = override.get("airway_points") or []
        if len(airway_points) >= 3:
            polygon = MplPath(np.asarray(airway_points, dtype=float))
            is_airway = polygon.contains_points(np.column_stack([world_x, world_z]))

    is_right = is_right_side & ~is_airway
    is_left = (~is_right_side) & ~is_airway
    return is_left, is_right, is_airway


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


BASE_OUTNAME = 'Optimized_Reconstruction-MI'
FALLBACK_OUTNAME = BASE_OUTNAME + '_fallback'
outname = BASE_OUTNAME  # current effective output-folder name for this process - see resolve_outname()
# --------------------------------------------------------------------------
# End of code copied from the notebook.
# --------------------------------------------------------------------------


def resolve_outname(main_dir):
    """Which output-folder name this main_dir's steps should read/write,
    for a run that doesn't include "recon" (e.g. re-running just
    Segment/Analysis later on a session that used the fallback timing
    template) - auto-detect which folder actually exists on disk,
    preferring the primary name. Matches the user's ask that "all the
    analysis should look for our output folder, if it doesn't exist then
    look for the _fallback folder". When "recon" IS in this run's steps,
    MI_reconstruction decides outname itself instead, per-session, from
    whether that session's own timing actually needed the fallback."""
    if os.path.isdir(os.path.join(main_dir, BASE_OUTNAME)):
        return BASE_OUTNAME
    if os.path.isdir(os.path.join(main_dir, FALLBACK_OUTNAME)):
        return FALLBACK_OUTNAME
    return BASE_OUTNAME


def _step_done(main_dir, step, **details):
    """Marks a step done for the GUI and records, in the session folder's
    fmig_code_version.json, which code version produced its output."""
    log_done(step)
    record_code_version(main_dir, step, outname=outname, **details)


def run(main_dir, steps, threshold=0.5, announce_done=True, use_fallback_timing=False):
    """announce_done=False suppresses the trailing "===ALL_DONE===" marker -
    used by run_group(), which calls this once per session, so the GUI
    doesn't mistake one session finishing for the whole group being done
    (run_group prints its own "===GROUP_ALL_DONE===" once every session in
    the group has actually run)."""
    global outname
    if "recon" not in steps:
        # No recon in this invocation, so nothing will call MI_reconstruction
        # to (re-)decide outname - resolve it from what's already on disk.
        outname = resolve_outname(main_dir)
        if outname != BASE_OUTNAME:
            print(f"Using fallback output folder: {outname}", flush=True)

    if "recon" in steps:
        log_step("recon")
        try:
            MI_reconstruction(main_dir, threshold=threshold, use_fallback_timing=use_fallback_timing)
        except DegenerateTimingError as e:
            # Distinct marker (not just ===STEP_FAILED===) so the GUI can
            # recognize this specific case and offer to retry with the
            # fallback timing template instead of just reporting a failure.
            print(f"===TIMING_FALLBACK_NEEDED=== {e}", flush=True)
            log_failed("recon", e)
            return 1
        except Exception as e:
            log_failed("recon", e)
            return 1
        if outname != BASE_OUTNAME:
            print(f"Using fallback output folder: {outname}", flush=True)
        _step_done(main_dir, "recon", threshold=threshold, use_fallback_timing=use_fallback_timing)

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
        _step_done(main_dir, "segment")

    if "segment_lr" in steps:
        log_step("segment_lr")
        try:
            for p in segment_lr_step(main_dir, outname):
                log_artifact(p)
        except Exception as e:
            log_failed("segment_lr", e)
            return 1
        _step_done(main_dir, "segment_lr")

    if "analysis" in steps:
        log_step("analysis")
        try:
            get_maps(main_dir, outname=outname, overwrite=False)
            # get_maps() only ever fills in FRC/TLC (needs Segment, not
            # Register) - TV/FV/J need a completed registration, which this
            # pipeline stores as a per-phase-pair R7_to_R0 folder rather
            # than the all-phase registered.npy/jacobian.npy get_maps()
            # looks for, so it silently skips them. Fill those three in
            # from the actual R7/R0 registration when it exists.
            try:
                get_maps_r7_r0(main_dir, outname, overwrite=False)
                print("TV/FV/J maps generated from the R7/R0 registration.")
            except FileNotFoundError as e:
                print(f"TV/FV/J maps skipped (no registration yet): {e}")
            log_artifact(os.path.join(main_dir, outname, 'Results', 'Maps', 'maps.png'))
            try:
                for key, p in get_maps_all_slice_panels(main_dir, outname, overwrite=False).items():
                    log_artifact(p)
            except FileNotFoundError as e:
                print(f"Single-slice panels skipped (no registration yet): {e}")

            # Rat-level view across sessions: rebuilt from every session's
            # own maps_panel_{EI,EE}_coronal.jpg each time Analysis runs,
            # so it always reflects whichever sessions have that panel so
            # far - a missing/failed panel here shouldn't fail this
            # session's own (already-succeeded) analysis, so it's a soft
            # warning per frame, not a step failure.
            rat_dir = os.path.dirname(os.path.normpath(main_dir))
            for frame in ("EI", "EE"):
                try:
                    rat_panel_path = build_rat_coronal_maps_panel(rat_dir, outname, frame=frame)
                    if rat_panel_path:
                        log_artifact(rat_panel_path)
                except Exception as e:
                    print(f"Rat-level {frame} coronal maps panel skipped: {e}")
        except Exception as e:
            log_failed("analysis", e)
            return 1
        _step_done(main_dir, "analysis")

    if "register" in steps:
        log_step("register")
        try:
            register_step(main_dir, outname, EE_toEI_only=True)
            get_difformation_fields(main_dir, outname)
        except Exception as e:
            log_failed("register", e)
            return 1
        _step_done(main_dir, "register")

    if "register_all_phases" in steps:
        # Every phase (1-15) registered onto R0/EI, not just the one phase
        # ("register" above only does phase 7 by default) - needed for a
        # per-voxel "expansion time" map (each voxel's own phase in the
        # cycle, from all 16 phases' Jacobians against a shared EI
        # reference) rather than a single EI/EE snapshot. Much slower than
        # "register" (15 registrations instead of 1), so it's its own
        # opt-in step rather than folded into "register".
        log_step("register_all_phases")
        try:
            failures = register_step(main_dir, outname, EE_toEI_only=False)
            if failures:
                print(f"Some phases failed to register to R0: {failures}")
        except Exception as e:
            log_failed("register_all_phases", e)
            return 1
        _step_done(main_dir, "register_all_phases")

    if "diaphragm" in steps:
        log_step("diaphragm")
        try:
            diaphragm_step(main_dir, outname)
        except Exception as e:
            log_failed("diaphragm", e)
            return 1
        _step_done(main_dir, "diaphragm")

    if "volume_analysis" in steps:
        log_step("volume_analysis")
        try:
            volume_analysis_step(main_dir, outname)
        except Exception as e:
            log_failed("volume_analysis", e)
            return 1
        _step_done(main_dir, "volume_analysis")

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


def compute_session_lung_volumes(main_dir, outname):
    """
    Air volume (mL) at this session's own true end-inspiration and
    end-expiration phases, split into left/right lungs at the mediastinal
    gap - the same direct, registration-independent method used throughout
    this conversation's manual FRC/TLC/TV checks. Needs only raw.npy/
    mask.npy/affine.npy (the Segment step), no Register step required, and
    uses each session's own detected EI/EE phase rather than assuming a
    fixed phase number - confirmed on real data that the true min/max-
    volume phase varies session to session (e.g. one PhNd31 session's true
    EE was phase 11, another's was phase 10, neither is the pipeline's
    default registration phase 7).

    EI/EE are selected by whichever phase has the most/least whole-lung
    HU-weighted air content (the same quantity TLC/FRC/TV are computed
    from) - NOT by mask voxel count, which an earlier version of this
    function used as a proxy. The two don't always agree: confirmed on
    real data (PhNd29 2026-09-10_13h48) that the largest-mask phase (R13)
    actually had LESS air content (4.930 mL) than R15 (5.313 mL) - a
    bigger segmented mask isn't necessarily the most air-filled one
    (partial-volume/boundary tissue can inflate mask size without adding
    much air). Picking EI/EE by mask size but then measuring air content
    at that phase silently undercounted TLC (and therefore TV) whenever
    the two signals disagreed - on that session, by about 18%. Selecting
    both phases by air content itself keeps the selection criterion and
    the measured quantity consistent, so this can only raise (never
    lower) the measured TLC-FRC swing relative to the old mask-size method.

    Raises FileNotFoundError if the Segment step hasn't been run yet.
    """
    RESULTS_DIR = os.path.join(main_dir, outname, "Results")
    raw_path = os.path.join(RESULTS_DIR, "raw.npy")
    mask_path = os.path.join(RESULTS_DIR, "mask.npy")
    affine_path = os.path.join(RESULTS_DIR, "affine.npy")
    for p in (raw_path, mask_path, affine_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p} not found - run the Segment step first.")

    imgs = np.load(raw_path)
    masks = np.load(mask_path)
    affine = np.load(affine_path)
    zooms = np.abs(np.diag(affine))[:3]
    voxel_vol_mm3 = float(zooms[0] * zooms[1] * zooms[2])

    def air_ml(img, ai, aj, ak):
        return float((img[ai, aj, ak] / -1000.0).sum() * voxel_vol_mm3 / 1000.0)

    air_by_phase = np.array([
        air_ml(imgs[p], *np.where(masks[p].astype(bool))) for p in range(masks.shape[0])
    ])
    ei_phase, ee_phase = int(np.argmax(air_by_phase)), int(np.argmin(air_by_phase))
    override = load_lr_split_override(main_dir, outname)

    def phase_lr(phase_idx):
        mask = masks[phase_idx].astype(bool)
        ai, aj, ak = np.where(mask)
        img = imgs[phase_idx]
        if override is not None:
            # Manually-drawn split/airway lines (see the app's Left/Right
            # Split editor) - a bent, Y-dependent split boundary instead
            # of the automatic method's single global X threshold, and
            # airway voxels excluded from both sides rather than
            # arbitrarily assigned to whichever falls on one side of that
            # threshold.
            vox = np.stack([ai, aj, ak, np.ones_like(ai)], axis=1).astype(float)
            world = (affine @ vox.T).T[:, :3]
            is_left, is_right, _ = classify_lr_with_override(
                world[:, 0], world[:, 1], world[:, 2], override, affine)
        else:
            is_left, is_right, _ = _split_diaphragm_left_right(ai, mask.shape[0], affine)
        return (air_ml(img, ai[is_left], aj[is_left], ak[is_left]),
                air_ml(img, ai[is_right], aj[is_right], ak[is_right]))

    tlc_left, tlc_right = phase_lr(ei_phase)
    frc_left, frc_right = phase_lr(ee_phase)

    return dict(
        ei_phase=ei_phase, ee_phase=ee_phase,
        frc_left_ml=frc_left, frc_right_ml=frc_right,
        tlc_left_ml=tlc_left, tlc_right_ml=tlc_right,
        tv_left_ml=tlc_left - frc_left, tv_right_ml=tlc_right - frc_right,
    )


VOLUME_ANALYSIS_CSV_FIELDS = [
    "session", "ei_phase", "ee_phase",
    "frc_left_mL", "frc_right_mL", "frc_total_mL",
    "tlc_left_mL", "tlc_right_mL", "tlc_total_mL",
    "tv_left_mL", "tv_right_mL", "tv_total_mL",
]


def save_lr_split_panel(mask, affine, out_path, title="", override=None):
    """
    2-panel (coronal, axial) illustration of the left/right lung split:
    every mask voxel projected and colored left (blue) or right (red),
    with the split boundary marked - lets you visually verify the split
    actually landed in the right place for this session. Both panels
    project the *entire* 3D mask (every voxel, not one cross-section
    through the middle), same "flatten the whole volume" idea as the maps
    panels' single-slice projections.

    With no override, uses the automatic single-threshold method
    (_split_diaphragm_left_right) and draws it as a straight line. With
    an override (see load_lr_split_override), uses the manually-drawn,
    possibly-bent split line and the airway exclusion polygon instead
    (classify_lr_with_override), airway voxels shown in gray, and draws
    the actual drawn split line / polygon rather than a straight
    threshold.
    """
    ai, aj, ak = np.where(mask)
    vox = np.stack([ai, aj, ak, np.ones_like(ai)], axis=1).astype(float)
    world = (affine @ vox.T).T[:, :3]

    is_airway = np.zeros(len(ai), dtype=bool)
    if override is not None:
        is_left, is_right, is_airway = classify_lr_with_override(
            world[:, 0], world[:, 1], world[:, 2], override, affine)
        # The whole-volume line/polygon is only meaningful to draw when
        # it's actually what classify_lr_with_override used - once
        # per-slice refinements exist for a boundary, a single global
        # line/polygon would misrepresent it (the coloring above already
        # reflects the correct per-slice/nearest-slice result either way).
        split_line_world = np.array(override["split_points"]) if not override.get("split_slices") else None
        airway_line_world = (np.array(override["airway_points"])
                              if override.get("airway_points") and not override.get("airway_slices") else None)
        split_idx = None
    else:
        is_left, is_right, split_idx = _split_diaphragm_left_right(ai, mask.shape[0], affine)
        # split_idx is a voxel index along axis 0 (L/R) - convert to its
        # world X coordinate (affine's X row only depends on axis 0 for
        # these axis-aligned NIfTI affines) so the split line lands at the
        # same spot on the world-mm axes the scatter itself uses.
        split_world_x = float(affine[0, 0] * split_idx + affine[0, 3])
        split_line_world = None
        airway_line_world = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor="black")
    plane_hv = {"Coronal": (0, 2), "Axial": (0, 1)}
    for ax, (plane, (h_idx, v_idx)) in zip(axes, plane_hv.items()):
        ax.set_facecolor("black")
        ax.scatter(world[is_left, h_idx], world[is_left, v_idx], s=3, c="#4090ff", label="left")
        ax.scatter(world[is_right, h_idx], world[is_right, v_idx], s=3, c="#ff5040", label="right")
        if is_airway.any():
            ax.scatter(world[is_airway, h_idx], world[is_airway, v_idx], s=3, c="#888888", label="airway (excluded)")
        # The split line is drawn on the axial (X/Y) plane, the airway
        # line on the coronal (X/Z) plane - each panel shows only the
        # line drawn on it; the other panel shows the resulting
        # left/right/airway coloring without the line itself.
        if plane == "Axial" and split_line_world is not None:
            ax.plot(split_line_world[:, 0], split_line_world[:, 1], color="white",
                     linestyle="--", linewidth=1.5, marker="o", markersize=3, label="split")
        elif plane == "Coronal" and split_line_world is None:
            ax.axvline(split_world_x, color="white", linestyle="--", linewidth=1.5, label="split")
        if plane == "Coronal" and airway_line_world is not None and len(airway_line_world) >= 3:
            poly = np.vstack([airway_line_world, airway_line_world[:1]])  # close the polygon
            ax.plot(poly[:, 0], poly[:, 1], color="yellow",
                     linestyle="--", linewidth=1.5, marker="o", markersize=3, label="airway polygon")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(plane, color="white", fontsize=14)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=9, facecolor="black", labelcolor="white")
    fig.suptitle(title, color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="black")
    plt.close(fig)
    return split_idx


# Phases the deformation/FRC/TLC panels already key off (R0 = EI, R7 = the
# default registration EE) - the two phases segment_lr_step exports a
# labeled left/right mask for.
LR_MASK_PHASES = (0, 7)


def segment_lr_step(main_dir, outname):
    """
    Applies this session's left/right lung split - the manually-drawn
    override from the app's Left/Right Split editor if one has been saved
    (see load_lr_split_override), otherwise the automatic mediastinal-gap
    method (_split_diaphragm_left_right) - to the phase 0 and 7 masks, and
    writes a labeled CT_Phase_<n>_rtk_R<n>_mLR.nii.gz next to each
    (0=background/excluded airway, 1=left lung, 2=right lung). Returns the
    list of paths written.
    """
    override = load_lr_split_override(main_dir, outname)
    results_dir = os.path.join(main_dir, outname, "Results")
    written = []
    for phase in LR_MASK_PHASES:
        mpath = os.path.join(results_dir, f"CT_Phase_{phase}_rtk_R{phase}_m.nii.gz")
        if not os.path.isfile(mpath):
            continue
        img = nib.load(mpath)
        data = np.asarray(img.dataobj)
        affine = img.affine

        label = np.zeros(data.shape, dtype=np.uint8)
        ai, aj, ak = np.where(data > 0)
        if len(ai) > 0:
            if override is not None:
                vox = np.stack([ai, aj, ak, np.ones_like(ai)], axis=1).astype(float)
                world = (affine @ vox.T).T[:, :3]
                is_left, is_right, _ = classify_lr_with_override(
                    world[:, 0], world[:, 1], world[:, 2], override, affine)
            else:
                is_left, is_right, _ = _split_diaphragm_left_right(ai, data.shape[0], affine)
            label[ai[is_left], aj[is_left], ak[is_left]] = 1
            label[ai[is_right], aj[is_right], ak[is_right]] = 2

        out_path = mpath[:-len(".nii.gz")] + "LR.nii.gz"
        out_img = nib.Nifti1Image(label, affine, img.header)
        out_img.header.set_data_dtype(np.uint8)
        nib.save(out_img, out_path)
        written.append(out_path)
    return written


def volume_analysis_step(main_dir, outname):
    """
    Computes this session's own FRC/TLC/TV (left/right/total, mL) at its
    detected EI/EE phases (see compute_session_lung_volumes), and writes/
    updates this session's row in <rat_dir>/Analysis/volume_analysis.csv,
    where rat_dir is main_dir's parent directory - so the same CSV
    accumulates one row per session regardless of whether sessions are
    run one at a time (Single Dataset mode) or as part of a per-rat group
    run, and re-running a session just replaces its own row instead of
    duplicating it. Also saves a coronal+axial left/right split-check
    panel (see save_lr_split_panel) next to the CSV, named per session so
    each session's own panel doesn't overwrite another's - using the EI
    phase's mask, since it's the larger, most complete silhouette.
    """
    v = compute_session_lung_volumes(main_dir, outname)

    rat_dir = os.path.dirname(os.path.normpath(main_dir))
    session_name = os.path.basename(os.path.normpath(main_dir))
    analysis_dir = os.path.join(rat_dir, "Analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    csv_path = os.path.join(analysis_dir, "volume_analysis.csv")

    row = {
        "session": session_name,
        "ei_phase": v["ei_phase"], "ee_phase": v["ee_phase"],
        "frc_left_mL": round(v["frc_left_ml"], 4), "frc_right_mL": round(v["frc_right_ml"], 4),
        "frc_total_mL": round(v["frc_left_ml"] + v["frc_right_ml"], 4),
        "tlc_left_mL": round(v["tlc_left_ml"], 4), "tlc_right_mL": round(v["tlc_right_ml"], 4),
        "tlc_total_mL": round(v["tlc_left_ml"] + v["tlc_right_ml"], 4),
        "tv_left_mL": round(v["tv_left_ml"], 4), "tv_right_mL": round(v["tv_right_ml"], 4),
        "tv_total_mL": round(v["tv_left_ml"] + v["tv_right_ml"], 4),
    }

    existing_rows = []
    if os.path.isfile(csv_path):
        with open(csv_path, "r", newline="") as f:
            existing_rows = list(csv.DictReader(f))
    existing_rows = [r for r in existing_rows if r.get("session") != session_name]
    existing_rows.append(row)
    existing_rows.sort(key=lambda r: r["session"])  # session folders are named YYYY-MM-DD_HHhMM

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VOLUME_ANALYSIS_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)
    print(f"Volume analysis updated: {csv_path} (session {session_name}: EI=R{v['ei_phase']} EE=R{v['ee_phase']})")
    log_artifact(csv_path)

    masks = np.load(os.path.join(main_dir, outname, "Results", "mask.npy"))
    affine = np.load(os.path.join(main_dir, outname, "Results", "affine.npy"))
    override = load_lr_split_override(main_dir, outname)
    split_panel_path = os.path.join(analysis_dir, f"{session_name}_lr_split.jpg")
    save_lr_split_panel(masks[v["ei_phase"]].astype(bool), affine, split_panel_path,
                         title=f"{session_name} - left/right split (R{v['ei_phase']}, EI)",
                         override=override)
    log_artifact(split_panel_path)

    return csv_path


def register_phase0_to_baseline(baseline_dir, moving_dir, outname, antspath=None, force=False):
    """
    Registers phase 0 of moving_dir onto phase 0 of baseline_dir - a
    DIFFERENT SESSION of the same rat (not a phase within one session,
    which is what register_step/register_all_phases do) - and also warps
    baseline_dir's phase-0 mask onto moving_dir's original grid, for
    visual comparison against the raw moving scan. Ported from
    Desktop/Pipeline/"register to baseline.ipynb", adapted to this app's
    fixed `outname` and CT_Phase_0_rtk_R0(_m).nii.gz naming instead of
    that notebook's generic "*_R0.nii.gz" glob.

    yi2.sh needs both images in one folder, and every session here reuses
    the same "CT_Phase_0_rtk_R0" filename, so phase-0 image+mask from
    each session are staged into a scratch folder under distinct,
    traceable names (the session's own date/time folder name) before
    calling it.

    moving_dir's R0 is the one that gets warped ("fixname" in yi2.sh
    terms); baseline_dir's R0 stays fixed and defines the output grid
    ("movname").

    Returns the output directory (as a string) containing the warp,
    Jacobian, warped image, and the baseline mask warped onto moving's
    grid.
    """
    antspath = Path(antspath) if antspath else DEFAULT_ANTSPATH
    baseline_dir, moving_dir = Path(baseline_dir), Path(moving_dir)
    baseline_label, moving_label = baseline_dir.name, moving_dir.name

    baseline_results = baseline_dir / outname / "Results"
    moving_results = moving_dir / outname / "Results"
    baseline_img, baseline_mask = baseline_results / "CT_Phase_0_rtk_R0.nii.gz", baseline_results / "CT_Phase_0_rtk_R0_m.nii.gz"
    moving_img, moving_mask = moving_results / "CT_Phase_0_rtk_R0.nii.gz", moving_results / "CT_Phase_0_rtk_R0_m.nii.gz"
    for p in (baseline_img, baseline_mask, moving_img, moving_mask):
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found - run Segment on this session first.")

    output_dir = moving_results / "Registered_to_Baseline" / f"R0_to_{baseline_label}"
    staging_dir = output_dir / "inputs"
    staging_dir.mkdir(parents=True, exist_ok=True)

    fixname = f"{moving_label}_R0"    # gets warped
    movname = f"{baseline_label}_R0"  # stays fixed, defines output grid

    def stage(src, dst):
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            return
        shutil.copy2(src, dst)

    stage(moving_img, staging_dir / f"{fixname}.nii.gz")
    stage(moving_mask, staging_dir / f"{fixname}_m.nii.gz")
    stage(baseline_img, staging_dir / f"{movname}.nii.gz")
    stage(baseline_mask, staging_dir / f"{movname}_m.nii.gz")

    if " " in str(staging_dir) or " " in str(antspath):
        raise ValueError("yi2.sh does not quote its paths internally, so these paths must not contain spaces.")

    warped_path = output_dir / f"{fixname}_To_{movname}_Warped.nii.gz"
    forward_warp = output_dir / f"{fixname}_To_{movname}_TotalWarp_Forward.nii.gz"
    baseline_mask_on_moving_grid = output_dir / f"{movname}_m_To_{fixname}_grid.nii.gz"

    if warped_path.is_file() and not force:
        print(f"Already registered, skipping ({warped_path.name} exists). Pass force=True to re-run.")
    else:
        if not (antspath / "antsRegistration.exe").is_file():
            raise FileNotFoundError(f"antsRegistration.exe not found under antspath: {antspath}")
        bash_exe = find_bash()
        print(f"Registering {fixname} (moving) -> {movname} (baseline)")
        ok = run_registration(bash_exe, antspath, fixname, movname, staging_dir, output_dir)
        if not ok:
            raise RuntimeError(f"Registration failed: {fixname} -> {movname}")
        print(f"Done. Warped image: {warped_path}")

    if baseline_mask_on_moving_grid.is_file() and not force:
        print(f"Baseline mask already warped onto moving grid, skipping ({baseline_mask_on_moving_grid.name} exists).")
    else:
        if not forward_warp.is_file():
            raise FileNotFoundError(
                f"Expected forward warp field not found (yi2.sh should have produced it): {forward_warp}")
        print(f"Warping baseline mask ({movname}_m) onto moving's original grid ({fixname})")
        cmd = [
            str(antspath / "antsApplyTransforms.exe"), "-d", "3",
            "-r", str(staging_dir / f"{fixname}.nii.gz"),
            "-i", str(staging_dir / f"{movname}_m.nii.gz"),
            "-t", str(forward_warp),
            "-n", "NearestNeighbor",
            "-o", str(baseline_mask_on_moving_grid),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"antsApplyTransforms failed (baseline mask onto moving grid):\n{result.stdout}\n{result.stderr}")
        print(f"Done. Baseline mask on moving grid: {baseline_mask_on_moving_grid}")

    return str(output_dir)


def register_baseline_for_rat(rat_dir, outname, sessions=None, antspath=None, force=False):
    """
    Runs register_phase0_to_baseline for every session of a rat except the
    first (chronologically) - the rat's own earliest session is the
    baseline, matching "register to baseline.ipynb"'s driver loop. Used by
    run_group's "register_to_baseline" step, which is per-RAT rather than
    per-session (it needs the whole rat's session list up front to know
    which one is the baseline), unlike every other step. `sessions`, if
    given, skips re-discovering them (run_group already has the list).
    Returns the list of output directories produced (one per moving
    session actually registered).
    """
    sessions = sessions if sessions is not None else get_rat_session_dirs(rat_dir)
    if len(sessions) < 2:
        print(f"Only {len(sessions)} session(s) found for {rat_dir} - need at least 2 to register to a baseline.")
        return []
    baseline_dir = sessions[0]
    print(f"Baseline: {baseline_dir}")
    out_dirs = []
    for moving_dir in sessions[1:]:
        print(f"Registering: {moving_dir}")
        out_dirs.append(register_phase0_to_baseline(
            baseline_dir, moving_dir, outname, antspath=antspath, force=force))
    return out_dirs


def run_group(rat_dirs, steps, threshold=0.5, use_fallback_timing=False):
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
                rc = run(session_dir, steps, threshold=threshold, announce_done=False,
                         use_fallback_timing=use_fallback_timing)
            except Exception as e:
                print(f"===SESSION_FAILED=== {session_dir}: {e}", flush=True)
                traceback.print_exc()
                rc = 1
            if rc == 0:
                print(f"===SESSION_DONE=== {session_dir}", flush=True)
            else:
                overall_rc = 1
                print(f"===SESSION_FAILED=== {session_dir}: step(s) failed, see log above", flush=True)

        if "register_to_baseline" in steps:
            # Per-RAT, not per-session (unlike every other step) - it needs
            # this rat's whole session list up front to know which one is
            # the baseline, so it runs once here after that rat's sessions
            # have all gone through Segment above, rather than inside
            # run()'s per-session dispatch.
            log_step("register_to_baseline")
            try:
                # Resolve explicitly from the baseline session's own folder,
                # rather than trusting the module-level outname - by this
                # point it holds whatever the last session processed above
                # left it as, which needn't match this rat's baseline.
                baseline_outname = resolve_outname(sessions[0]) if sessions else outname
                for d in register_baseline_for_rat(rat_dir, baseline_outname, sessions=sessions):
                    log_artifact(d)
                log_done("register_to_baseline")
                record_code_version(rat_dir, "register_to_baseline", outname=baseline_outname,
                                    sessions=[os.path.basename(os.path.normpath(s)) for s in sessions])
            except Exception as e:
                overall_rc = 1
                log_failed("register_to_baseline", e)

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
    parser.add_argument("--use-fallback-timing", action="store_true",
                         help="Permission to substitute the known-good timing template when a "
                              "session's own per-projection timestamps are degenerate, instead of "
                              "failing with DegenerateTimingError. Output for any session that "
                              "actually needed this goes to '<outname>_fallback' instead of "
                              "'<outname>'. Only pass this after the user has explicitly confirmed "
                              "it (the GUI shows a prompt when recon hits this).")
    args = parser.parse_args()
    print(f"FMIG code version: {get_code_version()}", flush=True)

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
        return run_group(args.rats, steps, threshold=threshold, use_fallback_timing=args.use_fallback_timing)

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
    return run(main_dir, steps, threshold=threshold, use_fallback_timing=args.use_fallback_timing)


if __name__ == "__main__":
    sys.exit(main())
