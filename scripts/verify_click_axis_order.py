"""
Determine the axis convention of lesion-clicks.json empirically.

Why this matters: prep_fdat.py read the clicks with the wrong JSON keys
(data["tumor"] / data["background"], which do not exist -- the real layout is
{"points": [{"point": [...], "name": "tumor"|"background"}]}), so every training
case got an all-zero click channel and every model we trained is click-blind.

Fixing the keys is not enough: prep_fdat.py also took the volume shape from
SimpleITK, whose array order is (z, y, x), while the official
generate_gaussian_heatmap documents coords as [x, y, z] and indexes the array
directly. Getting this wrong would scatter clicks to transposed positions -- so
we settle the convention with ground truth before committing to a retrain.

Test: tumor clicks are placed ON lesions, so under the correct permutation a
large fraction must land inside the GT mask (and background clicks outside it).

Run on PC-C:
  <autopetv-python> scripts/verify_click_axis_order.py
"""
import os
import json
import zipfile
import argparse
import itertools

import numpy as np
import nibabel as nib

FORK_DEFAULT = r"C:\Users\OkudaLab08\autoPETV"
CASE_DEFAULT = "psma_ffcaa75377465b37_2018-03-04"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", default=FORK_DEFAULT)
    ap.add_argument("--case", default=CASE_DEFAULT)
    args = ap.parse_args()

    gt_path = os.path.join(args.fork, "test", "labels", f"{args.case}.nii.gz")
    zip_path = os.path.join(args.fork, "nnunet-baseline", "lesion-scribbles.zip")

    gt = (nib.load(gt_path).get_fdata() > 0).astype(np.uint8)
    print(f"case      : {args.case}")
    print(f"GT shape  : {gt.shape} (nibabel order)   GT voxels: {int(gt.sum())}")

    with zipfile.ZipFile(zip_path) as z:
        entry = f"{args.case}_lesion-clicks.json"
        pts = json.loads(z.read(entry)).get("points", [])
    tumor = [p["point"] for p in pts if p.get("name") == "tumor"]
    bg = [p["point"] for p in pts if p.get("name") == "background"]
    print(f"clicks    : tumor={len(tumor)}  background={len(bg)}\n")

    hdr = f"{'permutation':<14} {'tumor in-bounds':>16} {'tumor ON lesion':>17} {'bg ON lesion':>14}"
    print(hdr)
    print("-" * len(hdr))

    best = None
    for perm in itertools.permutations(range(3)):
        def hits(coords):
            inb = on = 0
            for c in coords:
                idx = tuple(c[p] for p in perm)
                if all(0 <= idx[i] < gt.shape[i] for i in range(3)):
                    inb += 1
                    on += int(gt[idx] > 0)
            return inb, on

        t_in, t_on = hits(tumor)
        _, b_on = hits(bg)
        t_rate = t_on / len(tumor) if tumor else 0.0
        label = "(" + ",".join("xyz"[p] for p in perm) + ")"
        print(f"{label:<14} {t_in:>10d}/{len(tumor):<5d} {t_on:>11d}/{len(tumor):<5d} {b_on:>8d}/{len(bg):<5d}")
        # correct convention: tumor clicks land on lesion, background ones do not
        score = t_rate - (b_on / len(bg) if bg else 0.0)
        if best is None or score > best[0]:
            best = (score, label, t_on, len(tumor), b_on, len(bg))

    print("\n==== VERDICT ====")
    score, label, t_on, t_n, b_on, b_n = best
    print(f"best permutation : {label}")
    print(f"  tumor clicks on lesion      : {t_on}/{t_n}")
    print(f"  background clicks on lesion : {b_on}/{b_n}  (should be ~0)")
    if score > 0.5:
        print(f">> Convention confirmed: index the nibabel array as {label}.")
    else:
        print(">> INCONCLUSIVE -- no permutation puts tumor clicks on the lesion.")
        print("   Clicks may be in world/mm coordinates rather than voxel indices.")


if __name__ == "__main__":
    main()
