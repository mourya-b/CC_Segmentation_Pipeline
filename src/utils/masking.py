import numpy as np

def make_donut_mask(h, w, inner_frac=0.08, outer_frac=0.45):
    """
    Build a binary donut mask centered on the image.
    Keeps the annular vessel region, zeros the catheter center and the corners/background.

    inner_frac: radius of inner hole as fraction of min(h, w) — covers the catheter
    outer_frac: outer radius as fraction of min(h, w) — clips background

    Returns a (h, w) float32 mask with 1.0 inside the donut, 0.0 outside.
    """
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    r2 = (y - cy) ** 2 + (x - cx) ** 2
    r = min(h, w) / 2.0
    inner_r = inner_frac * 2 * r
    outer_r = outer_frac * 2 * r
    mask = (r2 >= inner_r ** 2) & (r2 <= outer_r ** 2)
    return mask.astype(np.float32)


def apply_donut_mask(img, inner_frac=0.08, outer_frac=0.45):
    """
    Apply donut mask to a 2D grayscale or 3D (H, W, C) image.
    Pixels outside the donut are set to 0.
    """
    h, w = img.shape[:2]
    mask = make_donut_mask(h, w, inner_frac, outer_frac)
    if img.ndim == 3:
        mask = mask[..., None]
    return (img * mask).astype(img.dtype)