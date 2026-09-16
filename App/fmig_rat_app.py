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
    ("analysis", "3. Analysis  (FRC / TLC maps)"),
    ("register", "4. Register  (phase registration + deformation field)"),
    ("diaphragm", "5. Diaphragm Motion  (descent estimate + GIF, needs Register)"),
]


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
        saved_steps = set(self.cfg.get("group_steps", [s[0] for s in STEPS]))
        for key, label in STEPS:
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

    def _selected_steps(self, step_vars):
        return [key for key, _ in STEPS if step_vars[key].get()]

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

            active_labels = self.step_labels
            cmd = [MI_ENV_PYTHON, "-u", DRIVER, main_dir,
                   "--steps", ",".join(steps), "--threshold", f"{threshold:g}"]
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

        self._active_step_labels = active_labels
        self._active_steps = steps
        for key, _ in STEPS:
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
