"""
recon_server.py - warm ITK/RTK worker for milabs_rtk_recon.py.

Prototype: pays the ~40s itk/rtk "first template instantiation" DLL-load cost
(see milabs_rtk_recon._warm_up) exactly once at startup, then reconstructs
phases for as many client jobs as arrive, reusing the same warm process
instead of the one-fresh-process-per-run behavior of recon_phase.bat.

RTK's CUDA FDK filter leaks ~0.8 GB of GPU memory per reconstruction, so a
single process can't run forever: this server tracks phases done and free
VRAM, and once either budget runs low it (a) hands off any remaining phases
of the *current* job to a subprocess running milabs_rtk_recon.py directly -
the same safe, already-proven chunking path recon_phase.bat uses today - and
(b) exits cleanly afterward. Run it under start_recon_server.bat, which
restarts it automatically whenever it exits, so recycling is transparent.

Must run under the system Python 3.10 that has itk/RTK installed (the same
interpreter recon_phase.bat uses) - NOT a conda env.

Usage:
    python recon_server.py [--port 8765] [--max-phases 8] [--vram-headroom-mb 1200]
"""
import argparse
import gc
import io
import re
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Listener
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recon_server_common import HOST, PORT, AUTHKEY, LEAK_PER_PHASE_MB, gpu_free_mb

import milabs_rtk_recon as mrr  # noqa: E402  (sets up CUDA dll dirs, imports itk on import)
import itk  # noqa: E402  (already loaded via mrr; re-imported here for clarity)
import numpy as np  # noqa: E402


def prep_phase(phase_dir, overrides, downsample=1):
    """CPU-only stage of reconstruct_phase: read TIFFs, compute line integrals,
    build geometry. Touches no CUDA/GPU state, so it's safe to run on a
    background thread while the GPU stage of a different phase is in flight."""
    paths = mrr.find_inputs(phase_dir)
    cfg = mrr.build_config(paths, overrides)
    angles = mrr.get_angles(paths["proj_log"], cfg["first_angle"])
    proj = mrr.load_line_integrals(paths["corr"], cfg["i0"], cfg["wpc"], downsample)
    geometry = mrr.build_geometry(cfg, angles)
    return paths, cfg, proj, geometry


def finish_phase(phase_dir, prepped, out_name, vol_scale, export_matrices):
    """GPU + CPU-finish stage of reconstruct_phase: FDK reconstruct, smooth,
    write NIfTI. Must run on the thread that owns the CUDA context (the main
    thread) - never call this from the prefetch thread."""
    paths, cfg, proj, geometry = prepped
    dim = [max(1, n // vol_scale) for n in cfg["dimension"]]
    spacing = float(cfg["spacing"]) * vol_scale
    origin = [o + (vol_scale - 1) / 2.0 * float(cfg["spacing"]) for o in cfg["origin"]]
    rtk_origin = [-(o + (n - 1) * spacing) for o, n in zip(origin, dim)]

    mu = mrr.fdk_reconstruct(geometry, proj, dim, spacing, rtk_origin,
                              hann=cfg["hann"], hardware=cfg.get("hardware", "cuda"))
    mu = mu[::-1, ::-1, ::-1]
    hu = mrr.smooth_and_scale(mu, spacing, float(cfg["gaussblr"]) / max(1, vol_scale), cfg["hu"])

    results = paths["results"]
    results.mkdir(exist_ok=True)
    scan = re.sub(r"^CT_|\.log$", "", paths["milabs_log"].name) if paths["milabs_log"] else Path(phase_dir).name
    name = out_name or f"CT_{scan}_rtk.nii"
    nii_origin = [o + (vol_scale - 1) / 2.0 * float(cfg["spacing"]) for o in cfg["niiorigin"]]
    mrr.write_nifti(hu, spacing, nii_origin, results / name)
    if export_matrices:
        mrr.export_geometry(geometry, proj, results, stem=f"CT_{scan}_rtk_geometry")
    print(f"wrote {results / name}")


class _TeeToConn(io.TextIOBase):
    """Mirrors writes to the real stream and to the client connection as log
    messages, so tqdm/print output shows up live on both ends.

    sys.stdout is process-global, and the pipelined path (see prep_phase) runs
    a background prefetch thread that also prints/tqdm.writes through it while
    the main thread is doing the same - multiprocessing.connection.Connection
    isn't safe for concurrent send() calls from multiple threads, so writes
    are serialized here to keep the socket framing from being corrupted."""

    def __init__(self, real_stream, conn):
        self.real_stream = real_stream
        self.conn = conn
        self._lock = threading.Lock()

    def write(self, s):
        if s:
            with self._lock:
                self.real_stream.write(s)
                self.real_stream.flush()
                try:
                    self.conn.send({"type": "log", "text": s})
                except Exception:
                    pass
        return len(s)

    def flush(self):
        self.real_stream.flush()


def run_overflow_subprocess(phase_dirs, args_ns, conn):
    """Delegate leftover phases to milabs_rtk_recon.py's own chunked-subprocess
    path - unmodified, proven logic, used only for whatever doesn't fit in the
    current warm process's safe VRAM/phase budget."""
    script = str(Path(__file__).resolve().parent / "milabs_rtk_recon.py")
    cmd = [sys.executable, "-u", script, *(str(d) for d in phase_dirs)]
    if args_ns.out_name:
        cmd += ["-o", args_ns.out_name]
    if args_ns.downsample != 1:
        cmd += ["--downsample", str(args_ns.downsample)]
    if args_ns.vol_scale != 1:
        cmd += ["--vol-scale", str(args_ns.vol_scale)]
    if args_ns.hann is not None:
        cmd += ["--hann", str(args_ns.hann)]
    if args_ns.gaussblr is not None:
        cmd += ["--gaussblr", str(args_ns.gaussblr)]
    if args_ns.no_matrices:
        cmd += ["--no-matrices"]

    conn.send({"type": "log", "text": f"\n[server] {len(phase_dirs)} phase(s) exceed this "
                                       f"warm process's safe VRAM budget - running them in a "
                                       f"fresh subprocess (pays the ~40s warm-up once)...\n"})
    proc = subprocess.Popen(cmd, cwd=str(Path(script).parent), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        conn.send({"type": "log", "text": line})
    return proc.wait()


def handle_job(conn, job, phases_done_this_process):
    """Returns (returncode, phases_done_this_process, should_recycle)."""
    root_dir = job["root_dir"]
    out_name = job.get("out_name")
    downsample = job.get("downsample", 1)
    vol_scale = job.get("vol_scale", 1)
    hann = job.get("hann")
    gaussblr = job.get("gaussblr")
    no_matrices = job.get("no_matrices", False)

    args_ns = argparse.Namespace(out_name=out_name, downsample=downsample, vol_scale=vol_scale,
                                  hann=hann, gaussblr=gaussblr, no_matrices=no_matrices)

    phases = mrr.find_phase_dirs(root_dir)
    if not phases:
        conn.send({"type": "error", "message": f"no ct-data/corr under {root_dir} or its Phase_* subdirs"})
        return 1, phases_done_this_process, False

    max_phases = job.get("max_phases_per_process", 8)
    headroom_mb = job.get("vram_headroom_mb", 1200)
    free_mb = gpu_free_mb()
    budget_by_count = max(0, max_phases - phases_done_this_process)
    if free_mb is not None:
        budget_by_vram = max(0, int(free_mb - headroom_mb) // LEAK_PER_PHASE_MB)
        budget = min(budget_by_count, budget_by_vram)
    else:
        budget = budget_by_count

    direct, overflow = phases[:budget], phases[budget:]

    old_stdout = sys.stdout
    sys.stdout = _TeeToConn(old_stdout, conn)
    failed = []
    overrides = {"hann": hann, "gaussblr": gaussblr}
    try:
        total = len(phases)
        with ThreadPoolExecutor(max_workers=1) as prefetch:
            # kick off phase 0's CPU-bound prep (TIFF read + line integrals) before
            # the loop starts, then keep prefetching phase i+1 while phase i's
            # GPU-bound FDK/smoothing runs on this thread - CPU and GPU are
            # different hardware, so this overlap is free, not contended.
            next_prep = prefetch.submit(prep_phase, direct[0], overrides, downsample) if direct else None
            for i, d in enumerate(direct, 1):
                print(f"\n=== [{i}/{total}] {d.name} (warm, in-process, pipelined) ===", flush=True)
                this_prep = next_prep
                # submit the next phase's prep unconditionally, *before* resolving this
                # one's result, so one phase's prep/finish failure can't stop the next
                # phase from being attempted (matches the old per-phase independence).
                if i < len(direct):
                    next_prep = prefetch.submit(prep_phase, direct[i], overrides, downsample)
                try:
                    prepped = this_prep.result()
                    finish_phase(d, prepped, out_name, vol_scale, not no_matrices)
                except Exception as e:
                    print(f"*** {d.name} FAILED: {e}", flush=True)
                    traceback.print_exc(file=sys.stdout)
                    failed.append(d.name)
                gc.collect()
                phases_done_this_process += 1
    finally:
        sys.stdout = old_stdout

    overflow_rc = 0
    if overflow:
        overflow_rc = run_overflow_subprocess(overflow, args_ns, conn)

    free_mb_after = gpu_free_mb()
    should_recycle = (
        phases_done_this_process >= max_phases
        or (free_mb_after is not None and free_mb_after < headroom_mb)
        or bool(overflow)  # the subprocess already paid a fresh warm-up; recycle for hygiene too
    )

    conn.send({"type": "log", "text": f"\n[server] phases run warm in-process: {len(direct)}, "
                                       f"handed to subprocess: {len(overflow)}, "
                                       f"free VRAM now: {free_mb_after} MiB\n"})
    returncode = (1 if failed else 0) or overflow_rc
    return returncode, phases_done_this_process, should_recycle


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--max-phases", type=int, default=8,
                     help="max phases to reconstruct in-process before recycling this server")
    ap.add_argument("--vram-headroom-mb", type=int, default=1200,
                     help="stop using this process once free VRAM drops below this")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[server] warming up ITK/RTK (one-time, ~40s) on pid {__import__('os').getpid()} ...",
          flush=True)
    mrr._warm_up()
    # Also force-load itk.image_from_array's lazy-loaded dependencies here, on the
    # main thread, before the server ever starts accepting jobs: the pipelined
    # in-process path (see prep_phase) calls this from a background prefetch
    # thread, and ITK's lazy SWIG-module loading isn't necessarily thread-safe
    # for a *first* call - so make sure it isn't a first call anymore.
    itk.image_from_array(np.zeros((2, 2, 2), dtype=np.float32))
    print(f"[server] ready in {time.time() - t0:.1f}s. HAVE_CUDA={mrr.HAVE_CUDA}. "
          f"Listening on {HOST}:{args.port}", flush=True)

    listener = Listener((HOST, args.port), authkey=AUTHKEY)
    phases_done_this_process = 0
    stop_requested = False
    try:
        while True:
            print(f"[server] waiting for a client ({phases_done_this_process}/"
                  f"{args.max_phases} phase budget used this process)...", flush=True)
            conn = listener.accept()
            recycle = False
            try:
                job = conn.recv()
                if job.get("cmd") == "ping":
                    conn.send({"type": "pong", "have_cuda": mrr.HAVE_CUDA,
                               "phases_done_this_process": phases_done_this_process})
                    continue
                if job.get("cmd") == "shutdown":
                    print("[server] shutdown requested - exiting (watchdog will not restart).", flush=True)
                    conn.send({"type": "bye"})
                    stop_requested = True
                    break
                job.setdefault("max_phases_per_process", args.max_phases)
                job.setdefault("vram_headroom_mb", args.vram_headroom_mb)
                rc, phases_done_this_process, recycle = handle_job(conn, job, phases_done_this_process)
                conn.send({"type": "done", "returncode": rc})
            except (EOFError, ConnectionResetError):
                pass
            except Exception as e:
                traceback.print_exc()
                try:
                    conn.send({"type": "error", "message": str(e)})
                except Exception:
                    pass
            finally:
                conn.close()
            if stop_requested:
                break
            if recycle:
                print("[server] recycling (VRAM/phase budget reached) - exiting so the "
                      "watchdog can start a fresh warm process.", flush=True)
                break
    finally:
        listener.close()

    if stop_requested:
        sys.exit(99)  # tells start_recon_server.bat's watchdog not to restart


if __name__ == "__main__":
    main()
