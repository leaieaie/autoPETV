"""Stage 0 diagnostic (winner-route insurance check).

Question: can the KNOWN-GOOD official baseline weights (which scored 0.7800 on
the leaderboard) segment OUR prepared Dataset998 fold-0 validation cases?

  * MEAN Dice >> 0.43  -> our data prep is sound; the ResEncM 0.5132 result was a
                          TRAINING issue (ResEnc compromise / epochs), safe to
                          invest days in the EDT + on-the-fly-click retrain.
  * MEAN Dice ~ 0.43   -> single-shot precomputed clicks cap any model on this
                          data; the real lever is interactive training (exactly
                          the winner recipe) rather than a data bug.
  * SHAPE MISMATCH / near-0 Dice -> data/geometry defect to fix BEFORE retraining.

Runs the official nnUNetPlans weights (no TTA) on N fold-0 val cases and reports
per-case + mean Dice against our labels. ~3-5 min, no submission slot used.

Run on PC-C:  conda run -n autopetv python scripts/stage0_diag.py
"""
import json
import os
import shutil
import subprocess

import numpy as np
import SimpleITK as sitk

RAW_BASE = r"D:\autopet\dataset"
DS = "Dataset998_AutoPETV"
PREP = r"D:\autopet\nnUNet\preprocessed\Dataset998_AutoPETV"
OFFICIAL_RESULTS = r"C:\Users\OkudaLab08\autoPETV\nnunet-baseline\nnUNet_results"
N = 15  # number of fold-0 validation cases to test

imagesTr = os.path.join(RAW_BASE, DS, "imagesTr")
labelsTr = os.path.join(RAW_BASE, DS, "labelsTr")
indir = r"D:\autopet\_stage0_in"
outdir = r"D:\autopet\_stage0_out"


def main():
    splits = json.load(open(os.path.join(PREP, "splits_final.json")))
    val_cases = splits[0]["val"][:N]
    print(f"[stage0] fold-0 val subset: {len(val_cases)} cases")

    for d in (indir, outdir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    for c in val_cases:
        for ch in ("0000", "0001", "0002", "0003"):
            src = os.path.join(imagesTr, f"{c}_{ch}.nii.gz")
            if not os.path.exists(src):
                print(f"[stage0] MISSING input channel: {src}")
                return
            shutil.copy(src, os.path.join(indir, f"{c}_{ch}.nii.gz"))

    env = dict(os.environ)
    env["nnUNet_results"] = OFFICIAL_RESULTS
    env["nnUNet_raw"] = RAW_BASE
    env["nnUNet_preprocessed"] = os.path.dirname(PREP)
    cmd = (
        f'nnUNetv2_predict -i "{indir}" -o "{outdir}" -d 998 -c 3d_fullres '
        f"-f 0 -tr nnUNetTrainer -p nnUNetPlans --disable_tta"
    )
    print(f"[stage0] predicting with OFFICIAL baseline weights:\n  {cmd}")
    subprocess.run(cmd, shell=True, check=True, env=env)

    def dice(a, b):
        a = a > 0
        b = b > 0
        s = int(a.sum()) + int(b.sum())
        return 1.0 if s == 0 else 2.0 * int((a & b).sum()) / s

    scores = []
    for c in val_cases:
        pp = os.path.join(outdir, f"{c}.nii.gz")
        gp = os.path.join(labelsTr, f"{c}.nii.gz")
        if not os.path.exists(pp):
            print(f"  {c}: NO PREDICTION written")
            continue
        p = sitk.GetArrayFromImage(sitk.ReadImage(pp))
        g = sitk.GetArrayFromImage(sitk.ReadImage(gp))
        if p.shape != g.shape:
            print(f"  {c}: SHAPE MISMATCH pred{p.shape} vs gt{g.shape} -> skip (geometry defect!)")
            continue
        d = dice(p, g)
        scores.append(d)
        print(f"  {c}: Dice={d:.4f}  (pred_vox={int((p>0).sum())}, gt_vox={int((g>0).sum())})")

    if scores:
        print(f"\n[stage0] MEAN Dice (official weights, our {len(scores)} val cases) = {np.mean(scores):.4f}")
    else:
        print("\n[stage0] no comparable cases -> investigate geometry/format")
    print("[stage0] >>0.43 => data OK, ResEncM training was the gap")
    print("[stage0] ~0.43 => single-shot clicks cap both -> interactive training (winner recipe) is the lever")


if __name__ == "__main__":
    main()
