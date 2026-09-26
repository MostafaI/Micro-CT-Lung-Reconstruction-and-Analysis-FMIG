import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

# find_lungs() reuses the UNet architecture and the exact preprocessing/inference
# helpers already built and tested in the Rat-lung-segmentation-from-xray project
# (model.py, real_inference.py), rather than duplicating that logic here. A
# runtime-only subset of that project (model.py, a trimmed real_inference.py, and
# the checkpoint find_lungs() loads) is vendored into this repo under vendor/ -
# see vendor/Rat-lung-segmentation-from-xray - so this file has no dependency
# outside the repo. The full project (training, dashboard, notebooks) still lives
# at Desktop/Pipeline/Rat-lung-segmentation-from-xray.
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(REPO_DIR, "vendor", "Rat-lung-segmentation-from-xray")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model import UNet
from real_inference import load_real_tiff_as_model_input, predict_with_model


def _clean_mask(pred_native, threshold=0.5, min_size_frac=0.05):
    """Binarize at `threshold`, then drop small spurious connected components
    (speckle noise) while keeping both lungs."""
    mask_bin = (pred_native > threshold).astype(np.uint8)
    labeled, n_components = ndimage.label(mask_bin)
    if n_components == 0:
        return mask_bin
    sizes = ndimage.sum(mask_bin, labeled, range(1, n_components + 1))
    keep_labels = np.where(sizes >= min_size_frac * sizes.max())[0] + 1
    return np.isin(labeled, keep_labels).astype(np.uint8)


def _diaphragm_curve(mask_bin):
    """Bottom-most lung pixel per column = the lung's inferior border, i.e. the
    diaphragm line. Columns with no lung at all (e.g. a gap over the mediastinum)
    are simply skipped, leaving a natural break in the curve."""
    xs, ys = [], []
    for x in range(mask_bin.shape[1]):
        rows = np.where(mask_bin[:, x] > 0)[0]
        if rows.size > 0:
            xs.append(x)
            ys.append(rows.max())
    return np.array(xs), np.array(ys)


def find_lungs(TIFF_PATH,
              CHECKPOINT = os.path.join(PROJECT_DIR, "checkpoints", "best_model_1.pt")):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    state = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model_input, display_img, raw_tiff = load_real_tiff_as_model_input(TIFF_PATH)
    # continuous [0,1] probability map -- NOT thresholded/binarized
    pred_prob = predict_with_model(model, model_input, device)

    # upscale the 256x256 probability map back to display_img's native resolution
    native_size = (display_img.shape[1], display_img.shape[0])  # PIL wants (width, height)
    pred_native = np.array(
        Image.fromarray(pred_prob, mode="F").resize(native_size, resample=Image.BILINEAR)
    )

    # mask_pred = _clean_mask(pred_native)
    diaphragm_x, diaphragm_y = _diaphragm_curve(pred_native > 0.5) #mask_pred)
    diaphragm_y_smooth = (
        ndimage.median_filter(diaphragm_y, size=9) if len(diaphragm_y) > 9 else diaphragm_y
    )
    diaphragm_height = float(diaphragm_y_smooth.mean()) if diaphragm_y_smooth.size > 0 else float("nan")

    return pred_native, diaphragm_x, diaphragm_y_smooth, diaphragm_height


if __name__ == "__main__":
    tiff_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\Data\PhNd5\2026-07-15_13h20\ct-data\corr\proj_000_0_00002103.tif"
    mask_pred, diaphragm_x, diaphragm_y_smooth, diaphragm_height = find_lungs(tiff_path)
    print(f"lung mask pixel count: {int(mask_pred.sum())}")
    print(f"diaphragm traced across {len(diaphragm_x)} of {mask_pred.shape[1]} image columns")
    print(f"mean diaphragm height (pixels from top): {diaphragm_height:.1f}")
