"""
nnUNetTrainer_OnTheFlyClicks
============================
Custom nnU-Net v2 (2.6.0) trainer for autoPET V interactive lesion
segmentation. It teaches the network to *respond to clicks* rather than to
segment from a single static prediction.

Two changes vs the stock nnUNetTrainer, applied together (LesionLocator style):

  1. On-the-fly clicks: instead of the fixed pre-computed click heatmaps baked
     into channels 2 (foreground) and 3 (background), we regenerate clicks from
     the (spatially augmented) ground truth patch on every iteration, with a
     random number of clicks (0..N). This exposes the network to the whole
     interaction trajectory (step 1 = few/no clicks ... step 6 = many clicks)
     instead of a single fixed click configuration.

  2. EDT encoding: clicks are encoded as a smooth exp(-d/tau) Euclidean distance
     field (value 1 at the click, decaying with distance), replacing the
     original single-voxel point marker that gave almost no spatial signal.

Deployment
----------
Drop this file into the installed nnU-Net package so it is discoverable by name,
e.g.:
    nnunetv2/training/nnUNetTrainer/variants/interactive/nnUNetTrainer_OnTheFlyClicks.py
Then train with:
    nnUNetv2_train 998 3d_fullres 0 -tr nnUNetTrainer_OnTheFlyClicks

Consistency requirements (see PC-C runbook):
  * plans.json channels 2 and 3 must use "NoNormalization".
  * inference (utils.save_click_heatmaps) must use the SAME EDT encoding
    (same tau / r_max). The reference implementation lives in
    nnunet-baseline/click_encoding.py; the encode function below is a byte-for
    -byte copy kept in sync with it.
"""

from typing import Union, Tuple, List

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_dilation

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import (
    DownsampleSegForDSTransform,
)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


# ---------------------------------------------------------------------------
# EDT encoding  (KEEP IN SYNC with nnunet-baseline/click_encoding.py)
# ---------------------------------------------------------------------------
DEFAULT_TAU = 3.0
DEFAULT_R_MAX = 12.0


def encode_clicks_edt(coords, shape, tau=DEFAULT_TAU, r_max=DEFAULT_R_MAX,
                      dtype=np.float32):
    """Encode click voxel coordinates into a smooth EDT heatmap in [0, 1]."""
    hm = np.zeros(shape, dtype=dtype)
    if coords is None or len(coords) == 0:
        return hm
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


# ---------------------------------------------------------------------------
# Click sampling from a ground-truth patch
# ---------------------------------------------------------------------------
def sample_clicks_from_seg(seg, rng, max_fg=10, max_bg=10,
                           bg_ring_iters=4, p_bg_ring=0.7, p_coldstart=0.1):
    """Sample foreground / background click voxels from a GT patch.

    Args:
        seg:   numpy (X, Y, Z), tumor where seg > 0.5.
        rng:   np.random.Generator.
    Returns:
        (fg_coords, bg_coords): two lists of [i, j, k].

    FG clicks are random interior tumor voxels (count 0..max_fg). BG clicks are
    biased toward a dilated tumor boundary ring (hard negatives that the
    interactive evaluation tends to correct in later steps) and otherwise drawn
    from arbitrary non-tumor voxels (count 0..max_bg). With prob p_coldstart we
    emit no clicks at all, covering the step-1 cold-start regime.
    """
    fg_coords, bg_coords = [], []
    tumor = seg > 0.5

    n_fg = int(rng.integers(0, max_fg + 1))
    n_bg = int(rng.integers(0, max_bg + 1))
    if rng.random() < p_coldstart:
        n_fg, n_bg = 0, 0

    # foreground: random interior tumor voxels
    if n_fg > 0 and tumor.any():
        tumor_idx = np.argwhere(tumor)
        take = min(n_fg, len(tumor_idx))
        pick = rng.integers(0, len(tumor_idx), size=take)
        fg_coords = tumor_idx[pick].tolist()

    # background: boundary ring (hard negatives) + arbitrary non-tumor
    if n_bg > 0:
        nontumor_idx = np.argwhere(~tumor)
        ring_idx = None
        if tumor.any():
            dil = binary_dilation(tumor, iterations=bg_ring_iters)
            ring = dil & (~tumor)
            if ring.any():
                ring_idx = np.argwhere(ring)
        for _ in range(n_bg):
            use_ring = ring_idx is not None and rng.random() < p_bg_ring
            pool = ring_idx if use_ring else nontumor_idx
            if pool is None or len(pool) == 0:
                break
            bg_coords.append(pool[rng.integers(0, len(pool))].tolist())

    return fg_coords, bg_coords


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------
class OnTheFlyClickTransform(BasicTransform):
    """Overwrite image channels 2/3 with freshly sampled EDT click heatmaps.

    Reads the (augmented) GT from the 'segmentation' entry and writes the
    'image' entry in place. Placed at the end of the augmentation pipeline so
    the click channels stay clean (they are not perturbed by the intensity
    augmentations), matching the clean heatmaps produced at inference time.
    """

    def __init__(self, fg_channel=2, bg_channel=3,
                 tau=DEFAULT_TAU, r_max=DEFAULT_R_MAX,
                 max_fg=10, max_bg=10, bg_ring_iters=4, p_bg_ring=0.7,
                 p_coldstart=0.1):
        super().__init__()
        self.fg_channel = fg_channel
        self.bg_channel = bg_channel
        self.tau = tau
        self.r_max = r_max
        self.max_fg = max_fg
        self.max_bg = max_bg
        self.bg_ring_iters = bg_ring_iters
        self.p_bg_ring = p_bg_ring
        self.p_coldstart = p_coldstart

    def apply(self, data_dict, **params):
        image = data_dict['image']            # torch (C, X, Y, Z)
        seg = data_dict['segmentation']       # torch (1, X, Y, Z)
        # per-worker/per-sample independent randomness
        rng = np.random.default_rng()
        seg_np = seg[0].detach().cpu().numpy()

        fg_coords, bg_coords = sample_clicks_from_seg(
            seg_np, rng, self.max_fg, self.max_bg,
            self.bg_ring_iters, self.p_bg_ring, self.p_coldstart,
        )
        shape = seg_np.shape
        fg_hm = encode_clicks_edt(fg_coords, shape, self.tau, self.r_max)
        bg_hm = encode_clicks_edt(bg_coords, shape, self.tau, self.r_max)

        image[self.fg_channel] = torch.from_numpy(fg_hm).to(image.dtype)
        image[self.bg_channel] = torch.from_numpy(bg_hm).to(image.dtype)
        data_dict['image'] = image
        return data_dict


# ---------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------
class nnUNetTrainer_OnTheFlyClicks(nnUNetTrainer):
    # tweakable without touching code below
    otf_fg_channel = 2
    otf_bg_channel = 3
    otf_tau = DEFAULT_TAU
    otf_r_max = DEFAULT_R_MAX
    otf_max_fg = 10
    otf_max_bg = 10
    otf_bg_ring_iters = 4
    otf_p_bg_ring = 0.7
    otf_p_coldstart = 0.1

    def _make_click_transform(self):
        return OnTheFlyClickTransform(
            fg_channel=self.otf_fg_channel, bg_channel=self.otf_bg_channel,
            tau=self.otf_tau, r_max=self.otf_r_max,
            max_fg=self.otf_max_fg, max_bg=self.otf_max_bg,
            bg_ring_iters=self.otf_bg_ring_iters, p_bg_ring=self.otf_p_bg_ring,
            p_coldstart=self.otf_p_coldstart,
        )

    @staticmethod
    def _inject(compose: ComposeTransforms, click_transform):
        """Insert the click transform just before deep-supervision downsampling
        (or at the very end if DS is disabled)."""
        transforms = compose.transforms
        insert_at = len(transforms)
        for i, t in enumerate(transforms):
            if isinstance(t, DownsampleSegForDSTransform):
                insert_at = i
                break
        transforms.insert(insert_at, click_transform)
        return compose

    def get_training_transforms(self, patch_size, rotation_for_DA,
                                deep_supervision_scales, mirror_axes,
                                do_dummy_2d_data_aug, use_mask_for_norm=None,
                                is_cascaded=False, foreground_labels=None,
                                regions=None, ignore_label=None) -> BasicTransform:
        compose = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded, foreground_labels=foreground_labels,
            regions=regions, ignore_label=ignore_label,
        )
        return self._inject(compose, self._make_click_transform())

    def get_validation_transforms(self, deep_supervision_scales,
                                  is_cascaded=False, foreground_labels=None,
                                  regions=None, ignore_label=None) -> BasicTransform:
        compose = nnUNetTrainer.get_validation_transforms(
            deep_supervision_scales, is_cascaded=is_cascaded,
            foreground_labels=foreground_labels, regions=regions,
            ignore_label=ignore_label,
        )
        # validation sees the same EDT-click input distribution as training so
        # the pseudo-dice reflects interactive performance
        return self._inject(compose, self._make_click_transform())
