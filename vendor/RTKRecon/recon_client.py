"""Client for recon_server.py. Import from any Python that can reach the
server's localhost port - no itk/rtk needed on the client side."""
import socket
from multiprocessing.connection import Client

from recon_server_common import HOST, PORT, AUTHKEY


def ping(host=HOST, port=PORT, timeout=1.5):
    """Returns the server's pong dict, or None if it's not reachable."""
    try:
        conn = Client((host, port), authkey=AUTHKEY)
    except (ConnectionRefusedError, socket.error, OSError):
        return None
    try:
        conn.send({"cmd": "ping"})
        if conn.poll(timeout):
            return conn.recv()
        return None
    except Exception:
        return None
    finally:
        conn.close()


def shutdown(host=HOST, port=PORT, timeout=2.0):
    """Asks the server to exit and tells the watchdog not to restart it.
    Returns True if a running server acknowledged, False if none was reachable."""
    try:
        conn = Client((host, port), authkey=AUTHKEY)
    except (ConnectionRefusedError, socket.error, OSError):
        return False
    try:
        conn.send({"cmd": "shutdown"})
        conn.poll(timeout)
        return True
    except Exception:
        return False
    finally:
        conn.close()


def run_recon_via_server(root_dir, on_line=print, host=HOST, port=PORT, connect_timeout=1.5, **job_kwargs):
    """Runs a reconstruction job on the warm server. Streams output lines to
    on_line(). Returns the process return code.

    Raises ConnectionRefusedError/OSError if the server isn't running -
    callers should catch that and fall back to the subprocess path.
    """
    try:
        conn = Client((host, port), authkey=AUTHKEY)
    except (ConnectionRefusedError, socket.error, OSError) as e:
        raise ConnectionRefusedError(f"recon server not reachable at {host}:{port}: {e}")

    try:
        job = {"cmd": "reconstruct", "root_dir": str(root_dir)}
        job.update(job_kwargs)
        conn.send(job)
        while True:
            msg = conn.recv()
            mtype = msg.get("type")
            if mtype == "log":
                on_line(msg["text"])
            elif mtype == "done":
                return msg["returncode"]
            elif mtype == "error":
                raise RuntimeError(f"recon server error: {msg['message']}")
            else:
                on_line(f"[unrecognized server message: {msg}]\n")
    finally:
        conn.close()
