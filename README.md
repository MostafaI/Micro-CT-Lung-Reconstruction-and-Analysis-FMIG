# Micro-CT-Lung-Reconstruction-and-Analysis-FMIG
A pipeline for reconstructing and analyzing preclinical micro-CT images for lung disorders that is currently used in the Functional and Metabolic Imaging Group (FMIG) at the University of Pennsylvania. 


Developed by PhD candidate: Mostafa Ismail

2021-2026

## Code version

The pipeline's code version lives in the `VERSION` file at the repo root (currently `v1.0`). Bump it there whenever a change affects results. The app shows it in its title bar and at the top of every run log.

Every time a step finishes, the pipeline records which version produced it in `fmig_code_version.json` inside that session's folder. Group-level registration writes the same file into the rat's folder. The file holds:

- `steps`: the latest entry for each step, showing which version made that step's current output.
- `history`: every entry, oldest first.

Each entry has the version, git commit, whether there were uncommitted changes, the time, the machine, and the step's settings.
