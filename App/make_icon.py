"""
One-off generator for App/lungs.ico - a simple flat lungs icon used for the
app window/taskbar icon and the Desktop shortcut. No external assets;
everything is drawn with PIL + numpy so it's fully reproducible offline.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os

S = 1024  # supersample size, downsampled at the end for clean anti-aliasing


def ellipse_mask(cx, cy, rx, ry, size=S):
    ys, xs = np.mgrid[0:size, 0:size]
    return (((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2) <= 1.0


def circle_mask(cx, cy, r, size=S):
    return ellipse_mask(cx, cy, r, r, size)


def thick_line_mask(p0, p1, width, size=S):
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    d.line([p0, p1], fill=255, width=width, joint="curve")
    d.ellipse([p0[0] - width / 2, p0[1] - width / 2, p0[0] + width / 2, p0[1] + width / 2], fill=255)
    d.ellipse([p1[0] - width / 2, p1[1] - width / 2, p1[0] + width / 2, p1[1] + width / 2], fill=255)
    return np.array(im) > 0


def dilate(mask, radius_px):
    im = Image.fromarray((mask * 255).astype(np.uint8))
    steps = max(1, radius_px // 8)
    for _ in range(steps):
        im = im.filter(ImageFilter.MaxFilter(9))
    return np.array(im) > 0


def build():
    cx = S / 2
    cy = S * 0.52

    # --- right lung (viewer's left): 3 lobes stacked, bulging outward ---
    right = (
        ellipse_mask(cx - 175, cy - 155, 150, 115)
        | ellipse_mask(cx - 195, cy - 20, 165, 130)
        | ellipse_mask(cx - 175, cy + 165, 190, 170)
    )

    # --- left lung (viewer's right): 2 lobes, smaller (leaves room for heart) ---
    left = (
        ellipse_mask(cx + 175, cy - 145, 135, 105)
        | ellipse_mask(cx + 195, cy + 100, 175, 195)
    )
    # cardiac notch: bite out of the lower-inner edge of the left lung
    notch = circle_mask(cx + 60, cy + 190, 145)
    left = left & ~notch

    # --- trachea + bronchi, unioned so there's no seam where they meet the lobes ---
    trachea = thick_line_mask((cx, S * 0.10), (cx, cy - 195), width=70)
    bronchus_r = thick_line_mask((cx, cy - 205), (cx - 150, cy - 130), width=58)
    bronchus_l = thick_line_mask((cx, cy - 205), (cx + 150, cy - 130), width=58)

    lungs = right | left | trachea | bronchus_r | bronchus_l

    # fissure lines (lobe boundaries) - subtle anatomical detail
    fissures = (
        thick_line_mask((cx - 330, cy - 70), (cx - 60, cy - 45), width=6)
        | thick_line_mask((cx - 300, cy + 90), (cx - 40, cy + 60), width=6)
        | thick_line_mask((cx + 40, cy - 30), (cx + 330, cy - 5), width=6)
    )
    fissures = fissures & lungs & ~dilate(np.zeros_like(lungs), 0)  # keep only inside lungs

    outline = dilate(lungs, 34) & ~lungs

    # --- colors ---
    LUNG = np.array([233, 121, 142, 255])       # warm pink
    LUNG_SHADE = np.array([214, 96, 120, 255])  # slightly deeper pink for lower lobes
    AIRWAY = np.array([238, 238, 240, 255])     # pale gray for trachea/bronchi
    OUTLINE = np.array([120, 40, 55, 255])      # deep maroon outline
    FISSURE = np.array([196, 76, 100, 200])

    img = np.zeros((S, S, 4), dtype=np.uint8)
    img[outline] = OUTLINE

    airway = trachea | bronchus_r | bronchus_l
    lobes_only = (right | left) & ~airway
    img[lobes_only] = LUNG
    # deepen the lower lobes a touch for visual depth
    lower = (ellipse_mask(cx - 175, cy + 165, 190, 170) | ellipse_mask(cx + 195, cy + 100, 175, 195)) & lobes_only
    img[lower] = LUNG_SHADE
    img[airway & lungs] = AIRWAY
    img[fissures] = FISSURE

    im = Image.fromarray(img, mode="RGBA")

    # rounded-square backplate so the icon reads clearly on any wallpaper/taskbar
    pad = 40
    plate = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    pd = ImageDraw.Draw(plate)
    radius = 190
    pd.rounded_rectangle([pad, pad, S - pad, S - pad], radius=radius, fill=(247, 249, 251, 255),
                          outline=(210, 216, 222, 255), width=8)
    out = Image.alpha_composite(plate, im)

    out = out.resize((256, 256), Image.LANCZOS)
    return out


def main():
    icon = build()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    ico_path = os.path.join(out_dir, "lungs.ico")
    png_path = os.path.join(out_dir, "lungs.png")
    icon.save(png_path)
    icon.save(ico_path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("Wrote", ico_path)
    print("Wrote", png_path)


if __name__ == "__main__":
    main()
