"""
Vendored, inference-only subset of real_inference.py from the
Rat-lung-segmentation-from-xray project (Desktop/Pipeline/Rat-lung-segmentation-from-xray).

Only `load_real_tiff_as_model_input` and `predict_with_model` are kept - the two
functions segment_from_projections.py's find_lungs() actually calls. The original
file's training-dashboard integration (save_result/update_real_inference/run_inference,
the `from dashboard import STATE_DIR` import, and the matplotlib-based comparison
plot) was dropped here since it depends on that project's Flask dashboard
(dashboard.py, static/, templates/), which isn't part of this vendored, runtime-only
copy. See the source project for the full version, including training and the
dashboard.

Runs the current best model on a real detector projection TIFF (not a simulated CT
projection). Real detector frames are raw transmission counts, so they're converted to
an attenuation-like image via -log(I/I_max) -- the same transform simulation_MI.ipynb
uses to compare real projections against simulated ones -- before being normalized and
resized the same way the training data was.
"""
import numpy as np
import torch
from PIL import Image

IMG_SIZE = 256


def load_real_tiff_as_model_input(tiff_path):
    raw = np.array(Image.open(tiff_path)).astype(np.float32)
    raw_max = raw.max()
    transmission = raw / raw_max if raw_max > 0 else raw
    # avoid log(0): transmission counts are always > 0 for real detector data
    attenuation = -np.log(np.clip(transmission, 1e-6, None))

    a_min, a_max = float(attenuation.min()), float(attenuation.max())
    if a_max - a_min < 1e-6:
        norm = np.zeros_like(attenuation, dtype=np.float32)
    else:
        norm = ((attenuation - a_min) / (a_max - a_min)).astype(np.float32)

    resized = np.array(
        Image.fromarray(norm, mode="F").resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    )
    return resized, norm, raw


def predict_with_model(model, model_input, device):
    """Run a forward pass with an already-loaded, already-in-memory model.

    Returns the raw sigmoid probability map (continuous, [0,1]) -- not thresholded.
    Use `binarize_mask` if you need a hard yes/no mask."""
    was_training = model.training
    model.eval()
    x = torch.from_numpy(model_input).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    model.train(was_training)
    return pred.astype(np.float32)


def binarize_mask(pred_prob, threshold=0.5):
    return (pred_prob > threshold).astype(np.float32)
