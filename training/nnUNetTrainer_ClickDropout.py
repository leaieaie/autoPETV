"""nnU-Net trainer that randomly hides the click channels during training.

Why this exists
---------------
Once the click channels were actually populated (prep_fdat.py had been writing
all-zero heatmaps for every case), the retrained model became strongly
click-responsive and beat the official baseline on lesion detection once clicks
arrived -- DMM 0.842 vs 0.778 at step 5. But at step 0, with no clicks, it
predicted 847 voxels and scored Dice 0.071, which sank the interactive AUC.

The cause is a shortcut in the data, measured over 300 sampled cases:

    tumor clicks present -> lesion-free labels:   0/189  (  0.0%)
    tumor clicks absent  -> lesion-free labels: 111/111  (100.0%)

The correlation is perfect, so "empty click channel" is a flawless predictor of
"no lesions" and the network has no reason to learn click-free detection. This
trainer breaks that correlation by showing lesion-positive cases with their
clicks removed, forcing the network to find lesions from CT+PET alone while
still learning to exploit clicks when they are given.

Why zero the whole channel
--------------------------
At inference, step 0 feeds click channels that are exactly zero everywhere, and
ZScore normalization of an all-zero channel leaves it at zero. Blanking the
whole channel therefore reproduces the step-0 input exactly. Thinning individual
clicks instead would leave the interpolation halo that resampling spreads around
each spike, leaking lesion positions in a way inference never does.

Usage
-----
Copy next to the stock trainers so nnU-Net can discover it:
    <env>/Lib/site-packages/nnunetv2/training/nnUNetTrainer/
then train with:
    nnUNetv2_train 998 3d_fullres 0 -tr nnUNetTrainer_ClickDropout
"""
import numpy as np
import torch

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import (
    DownsampleSegForDSTransform,
)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class ClickDropoutTransform(BasicTransform):
    """Blank both click channels for a random fraction of training samples."""

    def __init__(self, fg_channel=2, bg_channel=3, p_drop=0.5):
        super().__init__()
        self.fg_channel = fg_channel
        self.bg_channel = bg_channel
        self.p_drop = p_drop

    def apply(self, data_dict, **params):
        # fresh rng per call: dataloader workers would otherwise share a seed
        if np.random.default_rng().random() >= self.p_drop:
            return data_dict
        image = data_dict['image']            # torch (C, X, Y, Z)
        image[self.fg_channel] = 0
        image[self.bg_channel] = 0
        data_dict['image'] = image
        return data_dict


class nnUNetTrainer_ClickDropout(nnUNetTrainer):
    # half the samples are seen without clicks: enough to keep click-free
    # detection (that regime alone reached Dice 0.9202) while still leaving
    # plenty of clicked samples to learn from
    cd_fg_channel = 2
    cd_bg_channel = 3
    cd_p_drop = 0.5

    def _make_dropout_transform(self):
        return ClickDropoutTransform(
            fg_channel=self.cd_fg_channel,
            bg_channel=self.cd_bg_channel,
            p_drop=self.cd_p_drop,
        )

    @staticmethod
    def _inject(compose: ComposeTransforms, transform):
        """Insert just before deep-supervision downsampling (or at the end)."""
        transforms = compose.transforms
        insert_at = len(transforms)
        for i, t in enumerate(transforms):
            if isinstance(t, DownsampleSegForDSTransform):
                insert_at = i
                break
        transforms.insert(insert_at, transform)
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
        return self._inject(compose, self._make_dropout_transform())

    # validation deliberately keeps its clicks, so the pseudo dice stays
    # comparable with the 0.7733 of the run without dropout. Click-free
    # behaviour is judged by the 6-step diagnostic, not by this number.
