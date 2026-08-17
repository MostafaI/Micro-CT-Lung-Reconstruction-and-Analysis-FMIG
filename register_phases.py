#!/usr/bin/env python3
"""
Register every respiratory/cardiac phase (R1, R2, ...) in a reconstruction's
Results folder onto phase 0 (R0), by calling yi2.sh once per phase.

Expects the Results folder to contain, for each phase i, a pair of files
named "<prefix>_Ri.nii.gz" and "<prefix>_Ri_m.nii.gz" (image + mask), e.g.
    CT_2026-06-08_13h02_R0.nii.gz
    CT_2026-06-08_13h02_R0_m.nii.gz
    CT_2026-06-08_13h02_R1.nii.gz
    CT_2026-06-08_13h02_R1_m.nii.gz
    ...
which is what Optimized_Reconstruction/Results directories already look like.

Usage:
    python register_phases.py D:\\Data\\R45\\2026-06-08_13h02\\Optimized_Reconstruction\\Results
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
YI2_SH = SCRIPT_DIR / "yi2.sh"

# Where the native Windows ANTs 2.6.5 build was installed.
DEFAULT_ANTSPATH = Path(r"C:\Users\milabs\ants\ants-2.6.5\bin")

# Git for Windows' real bash, not the C:\Windows\system32\bash.exe WSL shim
# (which fails outright if no WSL distro is installed).
BASH_CANDIDATES = [
    Path(r"C:\Program Files\Git\bin\bash.exe"),
    Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
]

PHASE_RE = re.compile(r"^(.+)_R(\d+)\.nii\.gz$")


def find_bash():
    for candidate in BASH_CANDIDATES:
        if candidate.is_file():
            return candidate
    sys.exit("Could not find Git Bash. Checked: " + ", ".join(str(c) for c in BASH_CANDIDATES))


def discover_phases(results_dir):
    """Find "*_Rx.nii.gz" / "*_Rx_m.nii.gz" pairs and return {phase: prefix}.

    Each phase is matched independently on its own trailing "_Rx.nii.gz" /
    "_Rx_m.nii.gz", so it doesn't matter whether the part before "_Rx" is the
    same for every phase (e.g. "CT_2026-06-08_13h02") or embeds the phase
    number itself (e.g. "CT_Phase_7_rtk_R7.nii.gz").
    """
    by_phase = {}
    for entry in results_dir.iterdir():
        if not entry.is_file():
            continue
        m = PHASE_RE.match(entry.name)
        if not m:
            continue
        prefix, phase = m.group(1), int(m.group(2))
        mask = results_dir / f"{prefix}_R{phase}_m.nii.gz"
        if not mask.is_file():
            print(f"Warning: {entry.name} has no matching mask ({mask.name}), skipping")
            continue
        by_phase[phase] = prefix

    if not by_phase:
        sys.exit(f"No '*_R<n>.nii.gz' files found in {results_dir}")
    if 0 not in by_phase:
        sys.exit(f"Reference phase R0 not found in {results_dir}")

    return by_phase


def run_registration(bash_exe, antspath, fixname, movname, input_path, output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(bash_exe),
        str(YI2_SH),
        fixname,
        movname,
        str(input_path),
        str(output_path),
        str(antspath),
    ]
    print(f"\n=== {fixname} -> {movname} ===")
    result = subprocess.run(cmd)
    return result.returncode == 0


def register_all_phases(results_dir, antspath=DEFAULT_ANTSPATH, only_phases=None, force=False):
    """
    Register every phase in results_dir onto R0 via yi2.sh.

    results_dir: folder containing "<prefix>_Ri.nii.gz" / "<prefix>_Ri_m.nii.gz" pairs.
    only_phases: iterable of phase numbers to restrict to (e.g. [7] for just R7->R0);
                 None (default) registers every phase except R0.
    force: re-run phases whose output already exists (normally skipped).

    Returns the list of phases that failed (empty list = full success).
    """
    results_dir = Path(results_dir).resolve()
    antspath = Path(antspath)
    if not results_dir.is_dir():
        sys.exit(f"Results folder not found: {results_dir}")
    if " " in str(results_dir) or " " in str(antspath):
        sys.exit("yi2.sh does not quote its paths internally, so results_dir and antspath must not contain spaces.")
    if not YI2_SH.is_file():
        sys.exit(f"yi2.sh not found next to this script: {YI2_SH}")
    if not (antspath / "antsRegistration.exe").is_file():
        sys.exit(f"antsRegistration.exe not found under antspath: {antspath}")

    bash_exe = find_bash()
    by_phase = discover_phases(results_dir)
    targets = [p for p in sorted(by_phase) if p != 0]
    if only_phases is not None:
        wanted = set(only_phases)
        targets = [p for p in targets if p in wanted]
    movname = f"{by_phase[0]}_R0"
    print(f"Reference: R0 ({movname})   Phases to register: {targets}")

    registered_dir = results_dir / "Registered"
    registered_dir.mkdir(exist_ok=True)

    failures = []
    for phase in targets:
        fixname = f"{by_phase[phase]}_R{phase}"
        out_dir = registered_dir / f"R{phase}_to_R0"
        done_marker = out_dir / f"{fixname}_To_{movname}_Jacobian.nii.gz"
        if done_marker.is_file() and not force:
            print(f"\n=== R{phase} -> R0 already done, skipping ({done_marker.name} exists) ===")
            continue

        ok = run_registration(bash_exe, antspath, fixname, movname, results_dir, out_dir)
        if not ok:
            failures.append(phase)
            print(f"FAILED: R{phase} -> R0")

    print("\n=== Summary ===")
    print(f"{len(targets) - len(failures)}/{len(targets)} phases registered successfully")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_folder", type=Path, help=r"e.g. D:\Data\R45\2026-06-08_13h02\Optimized_Reconstruction\Results")
    parser.add_argument("--antspath", type=Path, default=DEFAULT_ANTSPATH, help="Folder containing antsRegistration.exe etc.")
    parser.add_argument("--phases", type=int, nargs="+", default=None, help="Only register these phase numbers, e.g. --phases 7")
    parser.add_argument("--force", action="store_true", help="Re-run phases whose output already exists")
    args = parser.parse_args()

    failures = register_all_phases(
        args.results_folder, antspath=args.antspath, only_phases=args.phases, force=args.force
    )
    if failures:
        print(f"Failed phases: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
