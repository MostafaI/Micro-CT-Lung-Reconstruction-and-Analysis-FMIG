"""
Code version tracking for the FMIG pipeline.

The version string lives in the VERSION file at the repo root - bump it
there (e.g. v1.0 -> v1.1) when a change affects results. Every session or
rat folder the pipeline writes into gets a fmig_code_version.json recording
which code version (plus git commit, when available) produced each step's
output, so results can always be traced back to the code that made them.
"""
import datetime
import json
import os
import platform
import subprocess
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION_FILE = os.path.join(REPO_DIR, "VERSION")
RECORD_NAME = "fmig_code_version.json"


def get_code_version():
    """The version string from VERSION, e.g. 'v1.0' ('unknown' if missing)."""
    try:
        with open(VERSION_FILE, "r", encoding="utf-8") as f:
            return f.read().strip() or "unknown"
    except OSError:
        return "unknown"


def _git(*args):
    try:
        out = subprocess.run(
            ["git", "-C", REPO_DIR, *args], capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


_code_info = None


def get_code_info():
    """Version plus best-effort git details (None where git isn't available).
    Computed once per process."""
    global _code_info
    if _code_info is None:
        status = _git("status", "--porcelain", "--untracked-files=no")
        _code_info = {
            "code_version": get_code_version(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_uncommitted_changes": (bool(status) if status is not None else None),
            "python": sys.version.split()[0],
            "machine": platform.node(),
        }
    return _code_info


def record_code_version(target_dir, step, **details):
    """Records that `step` was just (re)generated in target_dir by this code
    version. Updates target_dir/fmig_code_version.json:
      - "code_version": version of the most recent write
      - "steps": {step: latest entry} - which version made each step's output
      - "history": every entry, oldest first
    Never raises: a failure to write the record must not fail the pipeline."""
    path = os.path.join(target_dir, RECORD_NAME)
    entry = dict(get_code_info())
    entry["step"] = step
    entry["run_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    entry.update({k: v for k, v in details.items() if v is not None})
    try:
        record = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    record = json.load(f)
            except (OSError, ValueError):
                record = {}
        record["code_version"] = entry["code_version"]
        record["last_updated"] = entry["run_at"]
        record.setdefault("steps", {})[step] = entry
        record.setdefault("history", []).append(entry)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        os.replace(tmp, path)
    except OSError as e:
        print(f"Warning: could not write {path}: {e}", flush=True)
    return path
