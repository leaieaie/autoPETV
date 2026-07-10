"""
Shared EDT-based click encoding for autoPET V (interactive lesion segmentation).

This module defines the single source of truth for how a set of click voxel
coordinates is turned into a smooth heatmap channel. It is imported by:
  - inference:  nnunet-baseline/utils.py  (save_click_heatmaps)
  - training :  nnUNetTrainer_OnTheFlyClicks.py  (OnTheFlyClickTransform)

Encoding
--------
Given a set of click voxels, we compute the Euclidean distance transform d(x)
= distance from voxel x to the nearest click, then

    heatmap(x) = exp(-d(x) / tau),   set to 0 where d(x) > r_max

so the value is exactly 1.0 at a click and decays smoothly with distance.
Values are bounded in [0, 1]. This replaces the original single-voxel point
marker (sigma=0 Gaussian), which gave the network almost no spatial signal to
respond to.

IMPORTANT (train / inference consistency):
  * The FG/BG channels (index 2 and 3) must use NoNormalization in plans.json.
    A sparse heatmap z-scored would change scale with the number of clicks and
    break the correspondence between training and inference. With NoNormalization
    the raw [0, 1] EDT map is fed to the network in both regimes.
  * tau and r_max here MUST match the values used by the training transform.
"""

import numpy as np

# Encoding hyper-parameters (voxel units). Keep identical in trainer + inference.
DEFAULT_TAU = 3.0
DEFAULT_R_MAX = 12.0


def encode_clicks_edt(coords, shape, tau=DEFAULT_TAU, r_max=DEFAULT_R_MAX,
                      dtype=np.float32):
    """Encode click voxel coordinates into a smooth EDT heatmap.

    Args:
        coords: iterable of [i, j, k] integer voxel indices (may be empty).
        shape:  (X, Y, Z) shape of the output volume.
        tau:    decay length in voxels of the exp fall-off.
        r_max:  distances beyond this (voxels) are hard-clipped to 0.
        dtype:  output dtype.

    Returns:
        np.ndarray of `shape` with values in [0, 1]; all zeros if no valid click.
    """
    from scipy.ndimage import distance_transform_edt

    hm = np.zeros(shape, dtype=dtype)
    if coords is None or len(coords) == 0:
        return hm

    # scipy's EDT returns, for every non-zero voxel, the distance to the nearest
    # zero voxel. So we mark click voxels as 0 (the "features") and everything
    # else as 1; the EDT then gives distance-to-nearest-click everywhere.
    seed = np.ones(shape, dtype=np.uint8)
    any_valid = False
    for c in coords:
        i, j, k = int(c[0]), int(c[1]), int(c[2])
        if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
            seed[i, j, k] = 0
            any_valid = True
    if not any_valid:
        return hm

    d = distance_transform_edt(seed)
    hm = np.exp(-d / tau).astype(dtype)
    hm[d > r_max] = 0.0
    return hm
