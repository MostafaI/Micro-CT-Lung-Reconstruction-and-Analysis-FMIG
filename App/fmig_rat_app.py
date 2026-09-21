"""
FMIG Rat Reconstruction - desktop GUI front-end for Automated-FMIG-Rat.ipynb.

Pick a data folder, click Run, watch progress and preview images - no
notebook, no copy/pasting paths or code cells.

Runs pipeline_driver.py (next to this file) as a subprocess using the same
"mi-env" conda environment the notebook itself uses, and streams its output
live into this window.
"""
import json
import os
import subprocess
import sys
import threading
import queue
import time
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

# The Left/Right Split editor runs in-process (unlike every other feature
# here, which shells out to pipeline_driver.py in mi-env) - an interactive
# drawing tool needs live redraws on every click, and a subprocess round
# trip per click isn't workable. But `import pipeline_driver` alone takes
# ~12s (it pulls in the whole recon/ITK/ANTs stack at module level) -
# confirmed by timing it directly - so that import must NOT happen at this
# module's own import time, or every launch of this app would pay that
# 12s cost even for someone who never opens the editor. _load_editor_deps()
# below does the actual (slow, one-time, cached) import lazily, only when
# the editor is actually opened.
_editor_deps = None  # cache: None = not attempted, False = failed, dict = loaded


def _load_editor_deps():
    global _editor_deps
    if _editor_deps is not None:
        return _editor_deps
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        import pipeline_driver as pd
        _editor_deps = dict(np=np, FigureCanvasTkAgg=FigureCanvasTkAgg, Figure=Figure, pd=pd)
    except Exception as e:
        _editor_deps = False
        print(f"Left/Right Split editor dependencies failed to load: {e}", file=sys.stderr)
    return _editor_deps


APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(APP_DIR)  # repo root, holds recon_functions_all.py etc.
DRIVER = os.path.join(APP_DIR, "pipeline_driver.py")

# The notebook's kernel ("mi-env") - see Automated-FMIG-Rat.ipynb metadata
# and AppData/Roaming/jupyter/kernels/mi-env/kernel.json.
MI_ENV_PYTHON = r"C:\Users\milabs\.conda\envs\mi-env\python.exe"

CONFIG_DIR = os.path.join(os.environ.get("APPDATA", APP_DIR), "FMIGRatApp")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

# Projection segmentation threshold: pixels brighter than this (on a 0-1 scale)
# are dropped before the breathing signal is measured from the projections.
# 0.5 is the default for every folder; a per-folder override is stored under
# cfg["thresholds"][<normalized path>] once the user sets one via Check Threshold.
DEFAULT_THRESHOLD = 0.5

# RTKRecon is a separate, non-GitHub component that always lives under
# Desktop/Pipeline directly - not a sibling of this file, since this file
# ships inside the Micro-CT-Lung-Reconstruction-and-Analysis-FMIG repo.
RTKRECON_DIR = r"C:\Users\milabs\Desktop\Pipeline\RTKRecon"
START_SERVER_BAT = os.path.join(RTKRECON_DIR, "start_recon_server.bat")
SERVER_LOG = os.path.join(RTKRECON_DIR, "recon_server.log")
if RTKRECON_DIR not in sys.path:
    sys.path.insert(0, RTKRECON_DIR)

STEPS = [
    ("recon", "1. Reconstruction  (correct projections, breathing-gated recon)"),
    ("segment", "2. Segment  (compress, lung segmentation, cropped GIF)"),
    ("segment_lr", "3. Segment Left/Right lungs  (fix L/R split, save mLR masks, needs Segment)"),
    ("analysis", "4. Analysis  (FRC / TLC maps)"),
    ("register", "5. Register  (phase registration + deformation field)"),
    ("register_all_phases", "6. Register All Phases to EI  (every phase -> R0, needed for the "
                             "expansion-time map, slow)"),
    ("diaphragm", "7. Diaphragm Motion  (descent estimate + GIF, needs Register)"),
    ("volume_analysis", "8. Volume Analysis  (FRC/TLC/TV per lung, needs Segment)"),
]

# Group Analysis only - registering to a rat's own baseline session needs
# that rat's whole session list up front (to know which session IS the
# baseline), so it can't be decomposed into independent per-session work
# the way every step above can; it doesn't make sense for Single Dataset
# mode's one-session-at-a-time model. See pipeline_driver.run_group /
# register_baseline_for_rat.
GROUP_ONLY_STEPS = [
    ("register_to_baseline", "9. Register to Baseline  (align each later session's R0 onto the rat's "
                              "first session, needs Segment)"),
]
ALL_STEPS = STEPS + GROUP_ONLY_STEPS


def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg):
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f)
    except Exception:
        pass


class ThresholdDialog(tk.Toplevel):
    """Preview one projection with the segmentation threshold applied and let
    the user pick a value in (0, 1]. After it closes, self.result holds the
    chosen float, or None if the user cancelled."""

    PREVIEW_W = 520

    def __init__(self, parent, tiff_path, current=DEFAULT_THRESHOLD):
        super().__init__(parent)
        self.title("Check segmentation threshold")
        self.resizable(False, False)
        self.result = None

        try:
            import numpy as np
            self._np = np
            self._raw = np.asarray(Image.open(tiff_path)).astype(np.uint16) / 65535.0
        except Exception as e:
            messagebox.showerror("Check segmentation threshold",
                                 f"Could not load the projection image:\n{tiff_path}\n\n{e}",
                                 parent=parent)
            self.destroy()
            return

        h, w = self._raw.shape[:2]
        self._disp_size = (self.PREVIEW_W, max(1, int(round(h * self.PREVIEW_W / float(w)))))

        ttk.Label(self, text=os.path.basename(tiff_path), foreground="gray").pack(padx=10, pady=(10, 2))
        self._img_label = ttk.Label(self)
        self._img_label.pack(padx=10)
        ttk.Label(self,
                  text="Pixels brighter than the threshold are removed before the breathing "
                       "signal is measured from the projections.",
                  foreground="gray", wraplength=self.PREVIEW_W, justify="left").pack(padx=10, pady=(6, 0))

        row = ttk.Frame(self)
        row.pack(fill="x", padx=10, pady=(8, 0))
        ttk.Label(row, text="Threshold").pack(side="left")
        self._val_var = tk.StringVar()
        ttk.Label(row, textvariable=self._val_var, width=6, anchor="e").pack(side="right")
        self._scale = tk.Scale(self, from_=0.0, to=1.0, resolution=0.01,
                                orient="horizontal", showvalue=False, command=self._on_scale)
        self._scale.set(float(current))
        self._scale.pack(fill="x", padx=10)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=10, pady=10)
        ttk.Button(btns, text="Use this threshold", command=self._ok).pack(side="right")
        ttk.Button(btns, text="Cancel", command=self._cancel).pack(side="right", padx=6)

        self._render(float(current))
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.transient(parent)
        self.grab_set()
        self._scale.focus_set()
        self.wait_window(self)

    def _on_scale(self, _value):
        self._render(self._scale.get())

    def _render(self, t):
        np = self._np
        self._val_var.set(f"{t:.2f}")
        m = self._raw.copy()
        if t > 0:
            m[m > t] = 0.0
            disp = np.clip(m / t, 0.0, 1.0)
        else:
            disp = np.zeros_like(m)
        im = Image.fromarray((disp * 255).astype("uint8")).resize(self._disp_size)
        self._photo = ImageTk.PhotoImage(im)  # keep a reference alive
        self._img_label.config(image=self._photo)

    def _ok(self):
        self.result = round(float(self._scale.get()), 2)
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class LRSplitEditor(tk.Toplevel):
    """
    Interactive left/right lung split + airway exclusion editor. Shows the
    session's own EI-phase mask (largest lung volume across the 16-phase
    cycle - same choice save_lr_split_panel already makes). Two editing
    scopes (the "Scope" radio row), both saved into the same override:

      - "Whole volume" (coarse): draws ONE curve applied to every voxel
        regardless of slice - a left/right split line on the AXIAL (X/Y)
        projection, bending along the anterior-posterior axis, and an
        airway exclusion polygon on the CORONAL (X/Z) projection. This
        alone can't follow anatomy that changes in a more complex way
        than a single curve/polygon captures.
      - "This slice only" (fine): step through individual coronal slices
        (fixed Y, one at a time) with Prev/Next, and draw a split line
        and/or airway polygon specific to THAT slice, both in its own
        X/Z plane. Once any per-slice line exists for a boundary, every
        voxel uses whichever edited slice is nearest to it along Y (see
        pipeline_driver.classify_lr_with_override) - so touching up a
        handful of representative slices refines the untouched slices
        around them too, instead of requiring every slice to be edited
        by hand, and it replaces the whole-volume curve for that
        boundary entirely once used.

    Saved to Results/lr_split_override.json
    (pipeline_driver.save_lr_split_override); once saved,
    compute_session_lung_volumes and save_lr_split_panel use it
    automatically for this session instead of the automatic method. Save
    also immediately regenerates this session's phase 0/7 _mLR.nii.gz
    labeled masks (pipeline_driver.segment_lr_step).

    Opened from the "3. Segment Left/Right lungs" step (App._run /
    App._start_lr_editor_for_run) rather than a standalone button, so it
    runs in-process alongside the other steps' subprocess run instead of
    needing its own separate trigger.
    """

    # Per-axis stride for the interactive preview only - a plain modulo
    # (not the uniform-stride-over-flat-array fix used elsewhere in
    # pipeline_driver) is fine here since this never gets summed into a
    # reported volume, just displayed; the actual Save writes only the
    # drawn line points, and every volume computed from this session still
    # reads the full-resolution mask.
    DOWNSAMPLE_STRIDE = 3

    def __init__(self, parent, main_dir, outname, deps, on_close=None):
        super().__init__(parent)
        # deps: the dict from _load_editor_deps() - numpy/matplotlib/
        # pipeline_driver, loaded lazily by the caller (see
        # App._start_lr_editor_for_run) since importing pipeline_driver
        # alone takes ~12s and must not happen at this module's own import
        # time.
        self.np = deps["np"]
        self.pd = deps["pd"]
        self.Figure = deps["Figure"]
        self.FigureCanvasTkAgg = deps["FigureCanvasTkAgg"]
        np = self.np

        # Called with True (saved) or False (cancelled/failed to open) when
        # this dialog closes - lets the caller (the "3. Segment Left/Right
        # lungs" step, triggered from Run rather than a standalone button)
        # update its own step-status label instead of guessing.
        self.on_close = on_close

        self.main_dir = main_dir
        self.outname = outname
        session_name = os.path.basename(os.path.normpath(main_dir))
        self.title(f"Fix Left/Right Split - {session_name}")
        self.geometry("1150x700")

        mask_path = os.path.join(main_dir, outname, "Results", "mask.npy")
        affine_path = os.path.join(main_dir, outname, "Results", "affine.npy")
        if not os.path.isfile(mask_path) or not os.path.isfile(affine_path):
            messagebox.showerror(
                "Fix Left/Right Split",
                "This session needs the Segment step run first "
                "(mask.npy/affine.npy not found under Results\\).", parent=parent)
            self.destroy()
            if self.on_close:
                self.on_close(False)
            return

        masks = np.load(mask_path)
        self.affine = np.load(affine_path)
        sizes = masks.sum(axis=(1, 2, 3))
        self.ei_phase = int(np.argmax(sizes))
        self.full_mask = masks[self.ei_phase].astype(bool)

        ai, aj, ak = np.where(self.full_mask)
        keep = (ai % self.DOWNSAMPLE_STRIDE == 0)
        ai, aj, ak = ai[keep], aj[keep], ak[keep]
        vox = np.stack([ai, aj, ak, np.ones_like(ai)], axis=1).astype(float)
        self.world = (self.affine @ vox.T).T[:, :3]

        # Which array axis is the coronal-slicing (anterior-posterior, Y)
        # axis - see pipeline_driver._axis_for_world_row; this app already
        # assumes an axis-aligned affine everywhere else (L/R laterality,
        # the diaphragm region cache, etc).
        self.y_axis = int(np.argmax(np.abs(self.affine[1, :3])))
        counts_along_y = self.full_mask.sum(axis=tuple(a for a in range(3) if a != self.y_axis))
        valid_y = np.where(counts_along_y > 0)[0]
        self.y_min_idx = int(valid_y.min())
        self.y_max_idx = int(valid_y.max())
        self.current_slice_idx = int(np.argmax(counts_along_y))  # widest cross-section, a sensible start

        self.scope_var = tk.StringVar(value="whole")
        self.mode_var = tk.StringVar(value="split")
        self.split_points = []
        self.airway_points = []
        self.split_slices = {}   # {int Y-index: [[x_mm, z_mm], ...]}
        self.airway_slices = {}  # {int Y-index: [[x_mm, z_mm], ...]}
        existing = self.pd.load_lr_split_override(main_dir, outname)
        if existing:
            self.split_points = [list(p) for p in existing.get("split_points", [])]
            self.airway_points = [list(p) for p in existing.get("airway_points", [])]
            self.split_slices = {int(k): [list(p) for p in v]
                                  for k, v in (existing.get("split_slices") or {}).items()}
            self.airway_slices = {int(k): [list(p) for p in v]
                                   for k, v in (existing.get("airway_slices") or {}).items()}

        self._build_ui()
        self._redraw()

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.transient(parent)
        self.grab_set()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(top, text=f"Editing R{self.ei_phase} (EI phase, largest lung volume)",
                  foreground="gray").pack(side="left")

        scope_row = ttk.Frame(self)
        scope_row.pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(scope_row, text="Scope:").pack(side="left")
        ttk.Radiobutton(scope_row, text="Whole volume (coarse, one curve)", variable=self.scope_var,
                         value="whole", command=self._on_scope_change).pack(side="left", padx=(6, 12))
        ttk.Radiobutton(scope_row, text="This slice only (fine, one coronal slice at a time)",
                         variable=self.scope_var, value="slice", command=self._on_scope_change).pack(side="left")

        # Packed once, always visible, at this fixed position - toggling it
        # with pack_forget()/pack() later (when the scope radio changes)
        # would re-add it AFTER already-packed siblings (btns included),
        # so it'd render below the button row instead of staying put.
        # Disabling its buttons is scope-aware; its position isn't.
        slice_nav_row = ttk.Frame(self)
        slice_nav_row.pack(fill="x", padx=10, pady=(0, 4))
        self.prev_slice_btn = ttk.Button(slice_nav_row, text="<< Prev slice", command=self._prev_slice)
        self.prev_slice_btn.pack(side="left")
        self.next_slice_btn = ttk.Button(slice_nav_row, text="Next slice >>", command=self._next_slice)
        self.next_slice_btn.pack(side="left", padx=6)
        self.slice_label_var = tk.StringVar()
        ttk.Label(slice_nav_row, textvariable=self.slice_label_var, foreground="gray").pack(side="left", padx=10)

        mode_row = ttk.Frame(self)
        mode_row.pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(mode_row, text="Drawing:").pack(side="left")
        ttk.Radiobutton(mode_row, text="Left/Right split line (white)",
                         variable=self.mode_var, value="split", command=self._redraw).pack(
            side="left", padx=(6, 12))
        ttk.Radiobutton(mode_row, text="Airway exclusion polygon (orange)",
                         variable=self.mode_var, value="airway", command=self._redraw).pack(side="left")
        self.mode_hint_var = tk.StringVar()
        ttk.Label(mode_row, textvariable=self.mode_hint_var, foreground="gray").pack(side="left", padx=10)

        fig_frame = ttk.Frame(self)
        fig_frame.pack(fill="both", expand=True, padx=10)
        self.fig = self.Figure(figsize=(10, 5.2), dpi=100, facecolor="black")
        self.ax_coronal = self.fig.add_subplot(1, 2, 1)
        self.ax_axial = self.fig.add_subplot(1, 2, 2)
        self.canvas = self.FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=10, pady=10)
        ttk.Button(btns, text="Undo Last Point", command=self._undo).pack(side="left")
        ttk.Button(btns, text="Clear Split Line", command=self._clear_split).pack(side="left", padx=6)
        ttk.Button(btns, text="Clear Airway Line", command=self._clear_airway).pack(side="left", padx=6)
        ttk.Button(btns, text="Save", command=self._save).pack(side="right")
        ttk.Button(btns, text="Cancel", command=self._cancel).pack(side="right", padx=6)

        self._on_scope_change()

    def _on_scope_change(self):
        in_slice_scope = self.scope_var.get() == "slice"
        nav_state = "normal" if in_slice_scope else "disabled"
        self.prev_slice_btn.config(state=nav_state)
        self.next_slice_btn.config(state=nav_state)
        if in_slice_scope:
            self.mode_hint_var.set("Click the coronal (left) plot for both - split line can bend; "
                                    "airway needs >= 3 points.")
        else:
            self.slice_label_var.set("")
            self.mode_hint_var.set("Split: click the AXIAL (right) plot. "
                                    "Airway: click the CORONAL (left) plot, >= 3 points.")
        self._redraw()

    def _prev_slice(self):
        if self.current_slice_idx > self.y_min_idx:
            self.current_slice_idx -= 1
            self._redraw()

    def _next_slice(self):
        if self.current_slice_idx < self.y_max_idx:
            self.current_slice_idx += 1
            self._redraw()

    def _active_points(self):
        if self.scope_var.get() == "slice":
            d = self.split_slices if self.mode_var.get() == "split" else self.airway_slices
            return d.setdefault(self.current_slice_idx, [])
        return self.split_points if self.mode_var.get() == "split" else self.airway_points

    def _on_click(self, event):
        # Whole-volume scope: split points are clicked on the axial (X/Y)
        # plot, airway points on the coronal (X/Z) plot. Slice scope: both
        # are clicked on the coronal panel, which now shows just this one
        # slice's own X/Z plane instead of the whole-volume projection.
        if self.scope_var.get() == "slice":
            target_ax = self.ax_coronal
        else:
            target_ax = self.ax_axial if self.mode_var.get() == "split" else self.ax_coronal
        if event.inaxes is not target_ax or event.xdata is None or event.ydata is None:
            return
        self._active_points().append([float(event.xdata), float(event.ydata)])
        self._redraw()

    def _undo(self):
        pts = self._active_points()
        if pts:
            pts.pop()
            self._redraw()

    def _clear_split(self):
        if self.scope_var.get() == "slice":
            self.split_slices.pop(self.current_slice_idx, None)
        else:
            self.split_points = []
        self._redraw()

    def _clear_airway(self):
        if self.scope_var.get() == "slice":
            self.airway_slices.pop(self.current_slice_idx, None)
        else:
            self.airway_points = []
        self._redraw()

    def _build_override_dict(self):
        return dict(
            split_points=self.split_points,
            airway_points=self.airway_points,
            split_slices={str(k): v for k, v in self.split_slices.items()},
            airway_slices={str(k): v for k, v in self.airway_slices.items()},
        )

    def _classify_points(self, x, y, z):
        np = self.np
        have_split = len(self.split_points) >= 2 or len(self.split_slices) > 0
        if not have_split:
            zeros = np.zeros(len(x), dtype=bool)
            return zeros, zeros, zeros
        override = self._build_override_dict()
        return self.pd.classify_lr_with_override(x, y, z, override, self.affine)

    def _slice_world_y(self, idx):
        return float(self.affine[1, self.y_axis] * idx + self.affine[1, 3])

    def _slice_world_xz(self, idx):
        np = self.np
        slicer = [slice(None)] * 3
        slicer[self.y_axis] = idx
        plane = self.full_mask[tuple(slicer)]
        p, q = np.where(plane)
        other_axes = [a for a in range(3) if a != self.y_axis]
        vox = np.zeros((len(p), 4))
        vox[:, other_axes[0]] = p
        vox[:, other_axes[1]] = q
        vox[:, self.y_axis] = idx
        vox[:, 3] = 1
        world = (self.affine @ vox.T).T[:, :3]
        return world[:, 0], world[:, 2]

    def _scatter_classified(self, ax, hh, vv, is_left, is_right, is_airway, subtitle):
        ax.clear()
        ax.set_facecolor("black")
        unclassified = ~(is_left | is_right | is_airway)
        if unclassified.any():
            ax.scatter(hh[unclassified], vv[unclassified], s=2, c="#666666")
        if is_left.any():
            ax.scatter(hh[is_left], vv[is_left], s=2, c="#4090ff")
        if is_right.any():
            ax.scatter(hh[is_right], vv[is_right], s=2, c="#ff5040")
        if is_airway.any():
            ax.scatter(hh[is_airway], vv[is_airway], s=2, c="yellow")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(subtitle, color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("white")

    def _redraw(self):
        np = self.np
        x, y, z = self.world[:, 0], self.world[:, 1], self.world[:, 2]
        is_left, is_right, is_airway = self._classify_points(x, y, z)
        scope = self.scope_var.get()

        self._scatter_classified(
            self.ax_axial, x, y, is_left, is_right, is_airway,
            "Axial (click to add split points)" if scope == "whole" else "Axial (reference)")

        if scope == "slice":
            xs, zs = self._slice_world_xz(self.current_slice_idx)
            ys = np.full(len(xs), self._slice_world_y(self.current_slice_idx))
            sl, sr, sa = self._classify_points(xs, ys, zs)
            self._scatter_classified(self.ax_coronal, xs, zs, sl, sr, sa,
                                      f"Coronal slice {self.current_slice_idx} (click to draw)")
        else:
            self._scatter_classified(self.ax_coronal, x, z, is_left, is_right, is_airway,
                                      "Coronal (click to add airway points)")

        if scope == "whole":
            if self.split_points:
                pts = np.array(self.split_points)
                self.ax_axial.plot(pts[:, 0], pts[:, 1], color="white", linewidth=1.5,
                                     marker="o", markersize=4, zorder=5)
            if self.airway_points:
                pts = np.array(self.airway_points)
                if len(pts) >= 3:
                    pts = np.vstack([pts, pts[:1]])  # close the polygon
                self.ax_coronal.plot(pts[:, 0], pts[:, 1], color="orange", linewidth=1.5,
                                       marker="o", markersize=4, zorder=5)
        else:
            for other_idx in set(self.split_slices) | set(self.airway_slices):
                if other_idx != self.current_slice_idx:
                    self.ax_axial.axhline(self._slice_world_y(other_idx), color="#888800",
                                           linewidth=0.5, linestyle=":")
            self.ax_axial.axhline(self._slice_world_y(self.current_slice_idx), color="cyan",
                                   linewidth=1, linestyle=":")

            pts = self.split_slices.get(self.current_slice_idx)
            if pts:
                pts_arr = np.array(pts)
                self.ax_coronal.plot(pts_arr[:, 0], pts_arr[:, 1], color="white", linewidth=1.5,
                                       marker="o", markersize=4, zorder=5)
            pts2 = self.airway_slices.get(self.current_slice_idx)
            if pts2:
                pts2_arr = np.array(pts2)
                if len(pts2_arr) >= 3:
                    pts2_arr = np.vstack([pts2_arr, pts2_arr[:1]])
                self.ax_coronal.plot(pts2_arr[:, 0], pts2_arr[:, 1], color="orange", linewidth=1.5,
                                       marker="o", markersize=4, zorder=5)

            edited = "yes" if (self.current_slice_idx in self.split_slices
                                or self.current_slice_idx in self.airway_slices) else "no"
            self.slice_label_var.set(
                f"Slice {self.current_slice_idx} (Y-range {self.y_min_idx}-{self.y_max_idx}), "
                f"edited: {edited}")

        self.canvas.draw_idle()

    def _save(self):
        if len(self.split_points) < 2 and not self.split_slices:
            messagebox.showerror(
                "Fix Left/Right Split",
                "Draw at least 2 points for the left/right split line first "
                "(whole-volume scope, or at least one slice in slice scope).",
                parent=self)
            return
        path = self.pd.save_lr_split_override(
            self.main_dir, self.outname, self.split_points, self.airway_points,
            split_slices={str(k): v for k, v in self.split_slices.items()},
            airway_slices={str(k): v for k, v in self.airway_slices.items()})
        try:
            mlr_paths = self.pd.segment_lr_step(self.main_dir, self.outname)
            mlr_msg = f"\n\nSaved {len(mlr_paths)} labeled mask(s): " + ", ".join(
                os.path.basename(p) for p in mlr_paths) if mlr_paths else ""
        except Exception as e:
            mlr_msg = f"\n\n(Could not save the _mLR.nii.gz masks: {e})"
        messagebox.showinfo(
            "Fix Left/Right Split",
            f"Saved:\n{path}\n\nThis session's Volume Analysis step will use this split from now on."
            f"{mlr_msg}",
            parent=self)
        self.destroy()
        if self.on_close:
            self.on_close(True)

    def _cancel(self):
        self.destroy()
        if self.on_close:
            self.on_close(False)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FMIG Rat Reconstruction")
        self.geometry("980x720")
        self.minsize(820, 600)

        self.cfg = load_config()
        self.proc = None
        self.reader_thread = None
        self.out_queue = queue.Queue()
        self.running = False
        self.step_labels = {}
        self.thumb_refs = []  # keep PhotoImage refs alive
        self.last_artifacts = {}
        self._auto_start_attempted = False
        self._pending_lr_editor_dir = None

        # Group Analysis mode: run the same steps across every session found
        # under one or more rat directories (see pipeline_driver.get_rat_session_dirs
        # / run_group) instead of one already-picked dataset - same idea as
        # Desktop/Pipeline/Rat_all_dates_analysis.ipynb's per-rat loops.
        self.group_rats = list(self.cfg.get("group_rats", []))
        self.group_step_vars = {}
        self.group_step_labels = {}
        # Which step-label dict _handle_line updates for the run currently in
        # flight - set in _run() from the mode active at launch time, so
        # toggling the mode radio buttons (disabled while running anyway)
        # can't retarget a run already in progress.
        self._active_step_labels = self.step_labels
        self._active_steps = []

        icon_path = os.path.join(APP_DIR, "lungs.ico")
        if os.path.isfile(icon_path):
            try:
                self.iconbitmap(icon_path)
            except Exception:
                pass

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        mode_frame = ttk.Frame(self)
        mode_frame.pack(fill="x", **pad)
        ttk.Label(mode_frame, text="Mode:", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.mode_var = tk.StringVar(value=self.cfg.get("mode", "single"))
        self.single_radio = ttk.Radiobutton(mode_frame, text="Single Dataset", variable=self.mode_var,
                                             value="single", command=self._on_mode_change)
        self.single_radio.pack(side="left", padx=(10, 4))
        self.group_radio = ttk.Radiobutton(mode_frame, text="Group Analysis", variable=self.mode_var,
                                            value="group", command=self._on_mode_change)
        self.group_radio.pack(side="left", padx=4)

        # Both mode frames are built up front and swapped in/out of this
        # container via pack/pack_forget in _on_mode_change, rather than
        # rebuilt each toggle - keeps widget state (selections, checkboxes)
        # intact when switching back and forth.
        self.mode_container = ttk.Frame(self)
        self.mode_container.pack(fill="x")
        self.single_frame = ttk.Frame(self.mode_container)
        self.group_frame = ttk.Frame(self.mode_container)
        self._build_single_ui(self.single_frame, pad)
        self._build_group_ui(self.group_frame, pad)

        server_frame = ttk.Frame(self)
        server_frame.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(server_frame, text="Recon server:").pack(side="left")
        self.server_status_var = tk.StringVar(value="checking...")
        self.server_status_label = ttk.Label(server_frame, textvariable=self.server_status_var, foreground="gray")
        self.server_status_label.pack(side="left", padx=6)
        self.start_server_btn = ttk.Button(server_frame, text="Start Recon Server",
                                            command=self._start_recon_server)
        self.start_server_btn.pack(side="left", padx=6)
        self.stop_server_btn = ttk.Button(server_frame, text="Stop Server",
                                           command=self._stop_recon_server, state="disabled")
        self.stop_server_btn.pack(side="left", padx=6)
        view_log = ttk.Label(server_frame, text="View log", foreground="#08c", cursor="hand2")
        view_log.pack(side="left", padx=(0, 6))
        view_log.bind("<Button-1>", lambda e: self._open_artifact(SERVER_LOG))
        ttk.Label(server_frame, text="(skips the ~40s itk/rtk warm-up on every reconstruction run)",
                  foreground="gray").pack(side="left")
        self._poll_server_status()

        btns = ttk.Frame(self)
        btns.pack(fill="x", **pad)
        self.run_btn = ttk.Button(btns, text="Run", command=self._run)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.thresh_btn = ttk.Button(btns, text="Check Threshold", command=self._check_threshold)
        self.thresh_btn.pack(side="left", padx=6)
        ttk.Button(btns, text="Open Output Folder", command=self._open_output_folder).pack(side="left", padx=6)
        self.thresh_var = tk.StringVar()
        ttk.Label(btns, textvariable=self.thresh_var, foreground="gray").pack(side="left", padx=8)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(btns, textvariable=self.status_var, foreground="#0a5").pack(side="right")
        self._refresh_threshold_label()

        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill="x", padx=10)

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, **pad)

        log_frame = ttk.LabelFrame(body, text="Log")
        text_container = ttk.Frame(log_frame)
        text_container.pack(fill="both", expand=True)
        self.log_text = tk.Text(text_container, wrap="word", state="disabled", bg="#111", fg="#ddd",
                                 insertbackground="#ddd", font=("Consolas", 9))
        yscroll = ttk.Scrollbar(text_container, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=yscroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        body.add(log_frame, weight=3)

        preview_frame = ttk.LabelFrame(body, text="Preview")
        self.preview_canvas = tk.Canvas(preview_frame, highlightthickness=0)
        preview_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_canvas.yview)
        self.preview_canvas.configure(yscrollcommand=preview_scroll.set)
        self.preview_canvas.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        preview_scroll.pack(side="right", fill="y", pady=4)

        self.preview_container = ttk.Frame(self.preview_canvas)
        self._preview_window = self.preview_canvas.create_window((0, 0), window=self.preview_container, anchor="nw")
        self.preview_container.bind(
            "<Configure>",
            lambda e: self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all")))
        self.preview_canvas.bind(
            "<Configure>",
            lambda e: self.preview_canvas.itemconfigure(self._preview_window, width=e.width))
        self.preview_canvas.bind("<MouseWheel>", self._preview_mousewheel)
        body.add(preview_frame, weight=2)

        self._on_mode_change()

    def _build_single_ui(self, parent, pad):
        top = ttk.Frame(parent)
        top.pack(fill="x", **pad)
        ttk.Label(top, text="Data folder:", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.dir_var = tk.StringVar(value=self.cfg.get("last_dir", ""))
        entry = ttk.Entry(top, textvariable=self.dir_var, state="readonly")
        entry.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(top, text="Browse...", command=self._browse).pack(side="left")

        steps_frame = ttk.LabelFrame(parent, text="Steps to run")
        steps_frame.pack(fill="x", **pad)
        self.step_vars = {}
        saved_steps = set(self.cfg.get("steps", [s[0] for s in STEPS]))
        for key, label in STEPS:
            var = tk.BooleanVar(value=key in saved_steps)
            self.step_vars[key] = var
            row = ttk.Frame(steps_frame)
            row.pack(fill="x", padx=6, pady=2, anchor="w")
            ttk.Checkbutton(row, text=label, variable=var).pack(side="left")
            status = ttk.Label(row, text="", width=12, foreground="gray")
            status.pack(side="right")
            self.step_labels[key] = status

    def _build_group_ui(self, parent, pad):
        # Group Analysis: run the same steps across every session under one
        # or more rat directories (see pipeline_driver.get_rat_session_dirs /
        # run_group) - same idea as Rat_all_dates_analysis.ipynb's per-rat
        # loops, just driven from this GUI's existing subprocess machinery.
        rats_frame = ttk.LabelFrame(parent, text="Rats to include (every session under each rat runs in turn)")
        rats_frame.pack(fill="x", **pad)

        list_row = ttk.Frame(rats_frame)
        list_row.pack(fill="x", padx=6, pady=(4, 2))
        self.rats_listbox = tk.Listbox(list_row, height=4, selectmode="extended",
                                        bg="#111", fg="#ddd", highlightthickness=0)
        self.rats_listbox.pack(side="left", fill="both", expand=True)
        rats_scroll = ttk.Scrollbar(list_row, orient="vertical", command=self.rats_listbox.yview)
        self.rats_listbox.configure(yscrollcommand=rats_scroll.set)
        rats_scroll.pack(side="left", fill="y")

        rats_btns = ttk.Frame(rats_frame)
        rats_btns.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(rats_btns, text="Add Rat...", command=self._add_rat).pack(side="left")
        ttk.Button(rats_btns, text="Remove Selected", command=self._remove_selected_rats).pack(side="left", padx=6)
        self.group_sessions_var = tk.StringVar()
        ttk.Label(rats_btns, textvariable=self.group_sessions_var, foreground="gray").pack(side="left", padx=8)
        self._refresh_rats_listbox()

        steps_frame = ttk.LabelFrame(parent, text="Steps to run (same steps, every session)")
        steps_frame.pack(fill="x", **pad)
        saved_steps = set(self.cfg.get("group_steps", [s[0] for s in ALL_STEPS]))
        for key, label in ALL_STEPS:
            var = tk.BooleanVar(value=key in saved_steps)
            self.group_step_vars[key] = var
            row = ttk.Frame(steps_frame)
            row.pack(fill="x", padx=6, pady=2, anchor="w")
            ttk.Checkbutton(row, text=label, variable=var).pack(side="left")
            status = ttk.Label(row, text="", width=12, foreground="gray")
            status.pack(side="right")
            self.group_step_labels[key] = status

        thresh_row = ttk.Frame(parent)
        thresh_row.pack(fill="x", padx=16, pady=(0, 6))
        ttk.Label(thresh_row, text="Projection threshold (same value, every session):").pack(side="left")
        self.group_threshold_var = tk.StringVar(value=f"{self.cfg.get('group_threshold', DEFAULT_THRESHOLD):g}")
        ttk.Entry(thresh_row, textvariable=self.group_threshold_var, width=6).pack(side="left", padx=6)

    def _on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "single":
            self.group_frame.pack_forget()
            self.single_frame.pack(fill="x")
            self.thresh_btn.config(state="normal")
            self._refresh_threshold_label()
        else:
            self.single_frame.pack_forget()
            self.group_frame.pack(fill="x")
            self.thresh_btn.config(state="disabled")
            self.thresh_var.set("")
        self.cfg["mode"] = mode
        save_config(self.cfg)

    # -------------------------------------------------------------- actions
    def _browse(self):
        initial = self.dir_var.get() or self.cfg.get("browse_root") or r"D:\Data"
        if not os.path.isdir(initial):
            initial = os.path.expanduser("~")
        chosen = filedialog.askdirectory(title="Select data folder (contains ct-data\\corr)", initialdir=initial)
        if chosen:
            self.dir_var.set(os.path.normpath(chosen))
            self._refresh_threshold_label()

    def _open_output_folder(self):
        if self.mode_var.get() == "single":
            d = self.dir_var.get()
            if d and os.path.isdir(d):
                os.startfile(d)
            else:
                messagebox.showinfo("FMIG Rat Reconstruction", "Pick a data folder first.")
        else:
            if self.group_rats:
                os.startfile(self.group_rats[-1])
            else:
                messagebox.showinfo("FMIG Rat Reconstruction", "Add a rat folder first.")

    def _start_lr_editor_for_run(self, main_dir):
        # Triggered by the "3. Segment Left/Right lungs" step from _run()/
        # _handle_exit() rather than a standalone button - the editor is
        # inherently interactive (mouse clicks to draw), so it runs
        # in-process instead of going through the subprocess pipeline the
        # other steps use.
        if "segment_lr" in self.step_labels:
            self.step_labels["segment_lr"].config(text="editor...", foreground="#c80")

        # First open in this run of the app pays pipeline_driver's own
        # ~12s import cost (see _load_editor_deps) - show a wait cursor
        # and status message instead of a silent freeze. Cached after the
        # first call, so later opens are instant.
        self.status_var.set("Loading Left/Right Split editor (first time only)...")
        self.config(cursor="watch")
        self.update_idletasks()
        deps = _load_editor_deps()
        self.config(cursor="")
        self.status_var.set("Ready.")
        if not deps:
            messagebox.showerror(
                "FMIG Rat Reconstruction",
                "This feature needs numpy/matplotlib and pipeline_driver importable in the "
                "Python running this app (it should already be, since this app is normally "
                "launched with mi-env's own python). See the console for the exact error.")
            if "segment_lr" in self.step_labels:
                self.step_labels["segment_lr"].config(text="FAILED", foreground="#c00")
            return

        LRSplitEditor(self, main_dir, deps["pd"].outname, deps, on_close=self._on_lr_editor_closed)

    def _on_lr_editor_closed(self, saved):
        if "segment_lr" not in self.step_labels:
            return
        if saved:
            self.step_labels["segment_lr"].config(text="done", foreground="#0a5")
            self.status_var.set("Finished.")
        else:
            self.step_labels["segment_lr"].config(text="cancelled", foreground="gray")
            self.status_var.set("Ready.")

    def _selected_steps(self, step_vars):
        # ALL_STEPS covers both single-mode's step_vars (only STEPS keys)
        # and group-mode's group_step_vars (STEPS + GROUP_ONLY_STEPS) -
        # the "key in step_vars" guard is what makes one function safe for
        # both, rather than needing a separate version per mode.
        return [key for key, _ in ALL_STEPS if key in step_vars and step_vars[key].get()]

    # ---------------------------------------------------------- group mode
    def _add_rat(self):
        initial = self.cfg.get("browse_root") or r"D:\Data"
        if not os.path.isdir(initial):
            initial = os.path.expanduser("~")
        chosen = filedialog.askdirectory(title=r"Select a rat's data folder (e.g. D:\Data\PhNd7)",
                                          initialdir=initial)
        if not chosen:
            return
        chosen = os.path.normpath(chosen)
        if chosen not in self.group_rats:
            self.group_rats.append(chosen)
            self.cfg["group_rats"] = self.group_rats
            self.cfg["browse_root"] = os.path.dirname(chosen)
            save_config(self.cfg)
            self._refresh_rats_listbox()

    def _remove_selected_rats(self):
        sel = list(self.rats_listbox.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            del self.group_rats[idx]
        self.cfg["group_rats"] = self.group_rats
        save_config(self.cfg)
        self._refresh_rats_listbox()

    def _refresh_rats_listbox(self):
        self.rats_listbox.delete(0, "end")
        for r in self.group_rats:
            self.rats_listbox.insert("end", r)
        n = len(self.group_rats)
        self.group_sessions_var.set(f"{n} rat{'s' if n != 1 else ''} selected")

    # ---------------------------------------------------- projection threshold
    def _threshold_key(self, main_dir):
        return os.path.normcase(os.path.normpath(main_dir))

    def _threshold_for(self, main_dir):
        saved = self.cfg.get("thresholds", {})
        return float(saved.get(self._threshold_key(main_dir), DEFAULT_THRESHOLD))

    def _set_threshold_for(self, main_dir, value):
        self.cfg.setdefault("thresholds", {})[self._threshold_key(main_dir)] = float(value)
        save_config(self.cfg)

    def _refresh_threshold_label(self):
        main_dir = self.dir_var.get().strip()
        if not main_dir:
            self.thresh_var.set(f"threshold: {DEFAULT_THRESHOLD:g} (default)")
            return
        t = self._threshold_for(main_dir)
        custom = self._threshold_key(main_dir) in self.cfg.get("thresholds", {})
        self.thresh_var.set(f"threshold: {t:g}" + ("" if custom else " (default)"))

    def _find_projection_tif(self, corr_dir):
        preferred = os.path.join(corr_dir, "proj_000_0_000{:05d}.tif".format(5000))
        if os.path.isfile(preferred):
            return preferred
        cands = sorted(f for f in os.listdir(corr_dir)
                       if f.startswith("proj_000_0_000") and f.endswith(".tif"))
        if not cands:
            return None
        return os.path.join(corr_dir, cands[len(cands) // 2])

    def _check_threshold(self):
        main_dir = self.dir_var.get().strip()
        if not main_dir or not os.path.isdir(main_dir):
            messagebox.showerror("FMIG Rat Reconstruction", "Please choose a valid data folder first.")
            return
        if not HAVE_PIL:
            messagebox.showerror("FMIG Rat Reconstruction",
                                 "Pillow (PIL) is needed to preview the projection but is not "
                                 "installed in this Python environment.")
            return
        corr_dir = os.path.join(main_dir, "ct-data", "corr")
        if not os.path.isdir(corr_dir):
            messagebox.showerror("FMIG Rat Reconstruction",
                                 f"No projections folder found:\n{corr_dir}")
            return
        tiff_path = self._find_projection_tif(corr_dir)
        if not tiff_path:
            messagebox.showerror("FMIG Rat Reconstruction",
                                 f"No projection .tif files found in:\n{corr_dir}")
            return
        dlg = ThresholdDialog(self, tiff_path, current=self._threshold_for(main_dir))
        if dlg.result is not None:
            self._set_threshold_for(main_dir, dlg.result)
            self._refresh_threshold_label()
            self._log(f"[threshold] set to {dlg.result:g} for this folder\n")

    # ------------------------------------------------------- recon server ---
    def _poll_server_status(self):
        threading.Thread(target=self._check_server_status, daemon=True).start()
        self.after(15000, self._poll_server_status)

    def _check_server_status(self):
        try:
            import recon_client
            info = recon_client.ping()
        except Exception:
            info = None
        self.after(0, lambda: self._update_server_status(info))

    def _update_server_status(self, info):
        if info:
            phases = info.get("phases_done_this_process", 0)
            cuda = "CUDA" if info.get("have_cuda") else "CPU only"
            self.server_status_var.set(f"● warm ({cuda}, {phases} phase(s) done this process)")
            self.server_status_label.config(foreground="#0a5")
            self.start_server_btn.config(state="disabled")
            self.stop_server_btn.config(state="normal")
        else:
            self.server_status_var.set("○ not running (reconstruction will use a cold subprocess)")
            self.server_status_label.config(foreground="gray")
            self.start_server_btn.config(state="normal")
            self.stop_server_btn.config(state="disabled")
            if not self._auto_start_attempted:
                self._auto_start_attempted = True
                self._start_recon_server()

    def _start_recon_server(self):
        if not os.path.isfile(START_SERVER_BAT):
            messagebox.showerror("FMIG Rat Reconstruction", f"Not found:\n{START_SERVER_BAT}")
            return
        try:
            log_f = open(SERVER_LOG, "a")
            subprocess.Popen(
                [START_SERVER_BAT],
                cwd=RTKRECON_DIR,
                stdout=log_f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except Exception as e:
            messagebox.showerror("FMIG Rat Reconstruction", f"Failed to start recon server:\n{e}")
            return
        self.server_status_var.set("starting... (~40s warm-up, runs in the background - see 'View log')")
        self.server_status_label.config(foreground="#c80")
        self.start_server_btn.config(state="disabled")
        # the existing 15s poll loop (started in __init__) will pick up "warm" once ready

    def _stop_recon_server(self):
        self.stop_server_btn.config(state="disabled")

        def do_stop():
            try:
                import recon_client
                recon_client.shutdown()
            except Exception:
                pass
            self.after(500, self._poll_server_status_once)

        threading.Thread(target=do_stop, daemon=True).start()

    def _poll_server_status_once(self):
        threading.Thread(target=self._check_server_status, daemon=True).start()

    def _run(self):
        if self.running:
            return
        if not os.path.isfile(MI_ENV_PYTHON):
            messagebox.showerror(
                "FMIG Rat Reconstruction",
                f"Could not find the Python environment used by the notebook:\n{MI_ENV_PYTHON}\n\n"
                "Edit MI_ENV_PYTHON at the top of fmig_rat_app.py if it has moved.")
            return

        mode = self.mode_var.get()
        want_lr_editor = False
        if mode == "single":
            main_dir = self.dir_var.get().strip()
            if not main_dir or not os.path.isdir(main_dir):
                messagebox.showerror("FMIG Rat Reconstruction", "Please choose a valid data folder first.")
                return
            steps = self._selected_steps(self.step_vars)
            if not steps:
                messagebox.showerror("FMIG Rat Reconstruction", "Select at least one step to run.")
                return
            threshold = self._threshold_for(main_dir)

            self.cfg["last_dir"] = main_dir
            self.cfg["browse_root"] = os.path.dirname(main_dir)
            self.cfg["steps"] = steps
            save_config(self.cfg)

            # "Segment Left/Right lungs" is interactive (mouse clicks to
            # draw) so it can't run inside the headless subprocess the
            # other steps use - pull it out of the subprocess's step list
            # and open the editor in-process instead, once any subprocess
            # steps finish (see _handle_exit) or immediately if it's the
            # only step selected.
            want_lr_editor = "segment_lr" in steps
            subprocess_steps = [s for s in steps if s != "segment_lr"]
            self._pending_lr_editor_dir = main_dir if want_lr_editor else None

            if not subprocess_steps:
                if want_lr_editor:
                    self._pending_lr_editor_dir = None
                    self._start_lr_editor_for_run(main_dir)
                return

            active_labels = self.step_labels
            cmd = [MI_ENV_PYTHON, "-u", DRIVER, main_dir,
                   "--steps", ",".join(subprocess_steps), "--threshold", f"{threshold:g}"]
            steps = subprocess_steps  # what _launch_process below should actually track
        else:
            if not self.group_rats:
                messagebox.showerror("FMIG Rat Reconstruction", "Add at least one rat folder first.")
                return
            steps = self._selected_steps(self.group_step_vars)
            if not steps:
                messagebox.showerror("FMIG Rat Reconstruction", "Select at least one step to run.")
                return
            try:
                threshold = float(self.group_threshold_var.get())
            except ValueError:
                threshold = DEFAULT_THRESHOLD
            threshold = min(1.0, max(0.01, threshold))

            self.cfg["group_rats"] = self.group_rats
            self.cfg["group_steps"] = steps
            self.cfg["group_threshold"] = threshold
            save_config(self.cfg)

            active_labels = self.group_step_labels
            cmd = [MI_ENV_PYTHON, "-u", DRIVER]
            for rat_dir in self.group_rats:
                cmd += ["--rats", rat_dir]
            cmd += ["--steps", ",".join(steps), "--threshold", f"{threshold:g}"]

        self._launch_process(cmd, active_labels, steps)
        if want_lr_editor and "segment_lr" in active_labels:
            active_labels["segment_lr"].config(text="queued", foreground="gray")

    def _launch_process(self, cmd, active_labels, steps):
        self._active_step_labels = active_labels
        self._active_steps = steps
        for key, _ in ALL_STEPS:
            if key in active_labels:
                active_labels[key].config(text="", foreground="gray")
        for key in steps:
            active_labels[key].config(text="queued", foreground="gray")
        self._clear_log()
        self._clear_preview()
        self.last_artifacts = {}

        self._log(f"$ {' '.join(cmd)}\n")
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=REPO_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
        except Exception as e:
            messagebox.showerror("FMIG Rat Reconstruction", f"Failed to start pipeline:\n{e}")
            return

        self.running = True
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.single_radio.config(state="disabled")
        self.group_radio.config(state="disabled")
        self.status_var.set("Running...")
        self.progress.start(12)

        self.reader_thread = threading.Thread(target=self._read_process_output, daemon=True)
        self.reader_thread.start()

    def _stop(self):
        if not self.proc:
            return
        if not messagebox.askyesno("FMIG Rat Reconstruction", "Stop the running pipeline?"):
            return
        self._kill_process_tree()
        self.status_var.set("Stopped.")

    def _kill_process_tree(self):
        if self.proc and self.proc.poll() is None:
            try:
                subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                                capture_output=True)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def _on_close(self):
        if self.running:
            if not messagebox.askyesno("FMIG Rat Reconstruction",
                                        "A pipeline run is in progress. Quit anyway and stop it?"):
                return
            self._kill_process_tree()
        self.destroy()

    # -------------------------------------------------------------- worker
    def _read_process_output(self):
        try:
            for line in self.proc.stdout:
                self.out_queue.put(("line", line.rstrip("\n")))
        except Exception as e:
            self.out_queue.put(("line", f"[reader error] {e}"))
        finally:
            rc = self.proc.wait() if self.proc else -1
            self.out_queue.put(("exit", rc))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.out_queue.get_nowait()
                if kind == "line":
                    self._handle_line(payload)
                elif kind == "exit":
                    self._handle_exit(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _handle_line(self, line):
        self._log(line + "\n")
        labels = self._active_step_labels
        if line.startswith("===STEP=== "):
            key = line.split("===STEP=== ", 1)[1].strip()
            if key in labels:
                labels[key].config(text="running...", foreground="#c80")
        elif line.startswith("===STEP_DONE=== "):
            key = line.split("===STEP_DONE=== ", 1)[1].strip()
            if key in labels:
                labels[key].config(text="done", foreground="#0a5")
        elif line.startswith("===STEP_FAILED=== "):
            rest = line.split("===STEP_FAILED=== ", 1)[1]
            key = rest.split(":", 1)[0].strip()
            if key in labels:
                labels[key].config(text="FAILED", foreground="#c00")
        elif line.startswith("===ARTIFACT=== "):
            path = line.split("===ARTIFACT=== ", 1)[1].strip()
            self._add_artifact(path)
        elif line.startswith("===SESSION=== "):
            # Group mode: a new session is starting - reset this run's step
            # labels back to "queued" so a status left over from the
            # previous session ("done"/"FAILED") doesn't linger and read as
            # if it already applied to the new one.
            rest = line.split("===SESSION=== ", 1)[1].strip()
            self.status_var.set(f"Running: {rest}")
            for key in self._active_steps:
                if key in labels:
                    labels[key].config(text="queued", foreground="gray")
        elif line.startswith("===GROUP_ALL_DONE==="):
            self.status_var.set("Finished (group).")
        elif line.startswith("===ALL_DONE==="):
            self.status_var.set("Finished.")

    def _handle_exit(self, code):
        self.running = False
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.single_radio.config(state="normal")
        self.group_radio.config(state="normal")
        self.progress.stop()
        if code == 0:
            self.status_var.set("Finished successfully.")
        else:
            self.status_var.set(f"Stopped (exit code {code}).")
        self.proc = None

        pending_dir = self._pending_lr_editor_dir
        self._pending_lr_editor_dir = None
        if pending_dir:
            if code == 0:
                self._start_lr_editor_for_run(pending_dir)
            elif "segment_lr" in self.step_labels:
                self.step_labels["segment_lr"].config(text="skipped", foreground="gray")

    # --------------------------------------------------------------- utils
    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _log(self, text):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _clear_preview(self):
        for w in self.preview_container.winfo_children():
            w.destroy()
        self.thumb_refs.clear()

    def _preview_mousewheel(self, event):
        # Windows sends <MouseWheel> with event.delta in multiples of 120.
        self.preview_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_preview_mousewheel(self, widget):
        # Child labels (thumbnails/links) would otherwise swallow the wheel
        # event before it reaches the canvas, so bind on every descendant
        # too, not just the canvas itself.
        widget.bind("<MouseWheel>", self._preview_mousewheel)
        for child in widget.winfo_children():
            self._bind_preview_mousewheel(child)

    def _add_artifact(self, path):
        if not os.path.isfile(path):
            return
        self.last_artifacts[path] = time.time()
        row = ttk.Frame(self.preview_container)
        row.pack(fill="x", pady=4, anchor="n")
        name = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()

        if HAVE_PIL and ext in (".png", ".jpg", ".jpeg", ".gif"):
            try:
                img = Image.open(path)
                if ext == ".gif":
                    img.seek(0)
                img.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(img)
                self.thumb_refs.append(photo)
                lbl = ttk.Label(row, image=photo, cursor="hand2")
                lbl.pack()
                lbl.bind("<Button-1>", lambda e, p=path: self._open_artifact(p))
            except Exception:
                pass

        link = ttk.Label(row, text=name, foreground="#08c", cursor="hand2")
        link.pack()
        link.bind("<Button-1>", lambda e, p=path: self._open_artifact(p))
        self._bind_preview_mousewheel(row)

    def _open_artifact(self, path):
        try:
            os.startfile(path)
        except Exception:
            webbrowser.open(path)


if __name__ == "__main__":
    app = App()
    app.mainloop()
