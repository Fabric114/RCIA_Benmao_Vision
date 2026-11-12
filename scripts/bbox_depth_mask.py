import math

import numpy as np

MIN_VALID_DEPTH_M = 0.001


def _valid_depth(values):
    return values[np.isfinite(values) & (values > MIN_VALID_DEPTH_M)]


def bbox_depth_mask(depth_m, bbox_xyxy, center_frac=0.35, half_window_m=0.11):
    """Return an (H, W) bool mask of pixels inside the bbox whose depth is near the object.

    The reference depth is the median of valid depth in the central `center_frac`
    part of the bbox. If that region has no valid depth, the whole bbox is used.
    """
    H, W = depth_m.shape[:2]
    mask = np.zeros((H, W), dtype=bool)

    x1, y1, x2, y2 = bbox_xyxy
    c1 = max(0, int(math.floor(x1)))
    r1 = max(0, int(math.floor(y1)))
    c2 = min(W, int(math.ceil(x2)))
    r2 = min(H, int(math.ceil(y2)))
    if c2 <= c1 or r2 <= r1:
        return mask

    box = depth_m[r1:r2, c1:c2]
    box_valid = _valid_depth(box)
    if box_valid.size == 0:
        return mask

    bh, bw = r2 - r1, c2 - c1
    ch = max(1, int(round(bh * center_frac)))
    cw = max(1, int(round(bw * center_frac)))
    rs = r1 + (bh - ch) // 2
    cs = c1 + (bw - cw) // 2
    center_valid = _valid_depth(depth_m[rs:rs + ch, cs:cs + cw])
    ref = float(np.median(center_valid if center_valid.size else box_valid))

    keep = np.isfinite(box) & (box > MIN_VALID_DEPTH_M) & (np.abs(box - ref) <= half_window_m)
    mask[r1:r2, c1:c2] = keep
    return mask
