"""Shared constants/helpers for recon_server.py <-> recon_client.py.

Prototype scope: localhost-only, single user, no real auth needed - the
authkey below just satisfies multiprocessing.connection's requirement that
both ends agree on a key. Do not expose this port beyond localhost.
"""
import subprocess

HOST = "127.0.0.1"
PORT = 8765
AUTHKEY = b"milabs-fmig-recon-server-prototype-v1"

# Conservative estimate of the known per-phase GPU memory leak in RTK's CUDA
# FDK filter (see milabs_rtk_recon.py's --phases-per-process comment: ~0.8 GB
# per reconstruction). Used to decide how many more phases are safe to run in
# the current warm process before recycling it.
LEAK_PER_PHASE_MB = 900


def gpu_free_mb():
    """Free VRAM in MiB, or None if nvidia-smi isn't available."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None
