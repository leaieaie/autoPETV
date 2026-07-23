"""
Test why the click-fixed model collapses when given no clicks.

With clicks it now beats the official baseline on lesion detection (DMM 0.842 vs
0.778), but at step 0 -- no clicks -- it predicts 847 voxels and scores Dice
0.071, which drags the interactive AUC below baseline.

Hypothesis: 573 of 1611 training cases (35.6%) have no clicks in
lesion-clicks.json, and those are the lesion-free cases. If empty clicks and
empty labels coincide, the training data teaches the shortcut
"no clicks => no lesions", and predicting nothing at step 0 is exactly what the
model learned to do.

This checks the correlation directly: for cases with and without clicks, how
often is the label empty, and how large is it.

Run on PC-C:
  <autopetv-python> scripts/check_click_label_correlation.py
"""
import os
import json
import zipfile
import argparse

import numpy as np
import nibabel as nib

LABELS_DEFAULT = r"D:\autopet\dataset\Dataset998_AutoPETV\labelsTr"
ZIP_DEFAULT = r"C:\Users\OkudaLab08\autoPETV\nnunet-baseline\lesion-scribbles.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS_DEFAULT)
    ap.add_argument("--zip", dest="zip_path", default=ZIP_DEFAULT)
    ap.add_argument("--n", type=int, default=300, help="cases to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with zipfile.ZipFile(args.zip_path) as z:
        entries = {n: z.read(n) for n in z.namelist()}

    cases = sorted(f[:-len(".nii.gz")] for f in os.listdir(args.labels)
                   if f.endswith(".nii.gz"))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(cases), size=min(args.n, len(cases)), replace=False)
    picked = [cases[i] for i in sorted(idx)]
    print(f"labels found: {len(cases)}; sampling {len(picked)}\n")

    # buckets: has clicks / no clicks -> label empty count, label voxel sizes
    stats = {True: {"n": 0, "empty": 0, "vox": []},
             False: {"n": 0, "empty": 0, "vox": []}}

    for i, case in enumerate(picked, 1):
        key = f"{case}_lesion-clicks.json"
        if key not in entries:
            continue
        pts = json.loads(entries[key]).get("points", [])
        has_clicks = any(p.get("name") == "tumor" for p in pts)

        lab = np.asanyarray(nib.load(os.path.join(args.labels, f"{case}.nii.gz")).dataobj)
        vox = int((lab > 0).sum())

        s = stats[has_clicks]
        s["n"] += 1
        s["empty"] += (vox == 0)
        s["vox"].append(vox)
        if i % 50 == 0:
            print(f"  {i}/{len(picked)}")

    print("\n==== CORRELATION: clicks vs labels ====")
    for has, name in ((True, "HAS tumor clicks"), (False, "NO tumor clicks")):
        s = stats[has]
        if not s["n"]:
            print(f"{name:<18}: none sampled")
            continue
        v = np.array(s["vox"])
        pct = 100.0 * s["empty"] / s["n"]
        print(f"{name:<18}: n={s['n']:<4} label-empty={s['empty']:<4} ({pct:5.1f}%)  "
              f"median label vox={int(np.median(v))}")

    print("\n==== READING ====")
    no = stats[False]
    yes = stats[True]
    if no["n"] and yes["n"]:
        no_rate = no["empty"] / no["n"]
        yes_rate = yes["empty"] / yes["n"]
        if no_rate > 0.8 and yes_rate < 0.2:
            print(">> CONFIRMED: no clicks almost always means no lesions.")
            print(">> The model learned the shortcut 'empty click channel => predict nothing',")
            print("   which is why step 0 collapses. Fix: train with click dropout so")
            print("   lesion-positive cases are also seen with no clicks.")
        elif no_rate > 0.5:
            print(">> Partially confirmed: empty clicks skew strongly toward lesion-free cases.")
        else:
            print(">> NOT confirmed: cases without clicks still contain lesions, so the")
            print("   step-0 collapse has another cause.")


if __name__ == "__main__":
    main()
