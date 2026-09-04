"""
pipeline_driver.py

Headless, script form of Desktop/Pipeline/Automated-FMIG-Rat.ipynb.
Runs the exact same steps, in the exact same order, using the same functions
from the same modules the notebook uses - just without Jupyter, and with
plain-text progress markers on stdout so a GUI can follow along.

Usage:
    python pipeline_driver.py "D:\\Data\\Txk5\\2026-08-06_10h47" [--steps recon,segment,analysis,register]

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

STEP_ORDER = ["recon", "segment", "analysis", "register"]


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

    threshold = 0.5
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    im = get_image(corr_dir, slice_num=coronal_slice_num, normalize=True, threshold=.5)

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
            
def MI_reconstruction(main_dir):
    threshold = 0.5
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

    di, dj, dk = ai, aj, ak
    if DENSE_STRIDE > 1:
        dkeep = (di % DENSE_STRIDE == 0) & (dj % DENSE_STRIDE == 0) & (dk % DENSE_STRIDE == 0)
        di, dj, dk = di[dkeep], dj[dkeep], dk[dkeep]
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


outname = 'Optimized_Reconstruction-MI'
# --------------------------------------------------------------------------
# End of code copied from the notebook.
# --------------------------------------------------------------------------


def run(main_dir, steps):
    if "recon" in steps:
        log_step("recon")
        try:
            MI_reconstruction(main_dir)
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

    print("\n===ALL_DONE===", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run the FMIG rat reconstruction pipeline headlessly.")
    parser.add_argument("main_dir", help=r'Data folder, e.g. D:\Data\Txk5\2026-08-06_10h47')
    parser.add_argument("--steps", default=",".join(STEP_ORDER),
                         help="Comma-separated subset of: " + ",".join(STEP_ORDER))
    args = parser.parse_args()

    main_dir = args.main_dir
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEP_ORDER)
    if unknown:
        print(f"Unknown step(s): {unknown}. Valid steps: {STEP_ORDER}", file=sys.stderr)
        return 2
    if not os.path.isdir(main_dir):
        print(f"main_dir does not exist: {main_dir}", file=sys.stderr)
        return 2

    print(f"main_dir = {main_dir}")
    print(f"steps = {steps}")
    return run(main_dir, steps)


if __name__ == "__main__":
    sys.exit(main())
