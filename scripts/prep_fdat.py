"""
Assemble the FDAT (PSMA-FDG-PET-CT-Lesions_v2) dataset into nnU-Net 4-channel
format for autoPET V (Dataset998_AutoPETV).

Inputs are already nnU-Net-ish:
  <fdat>/imagesTr/<case>_0000.nii.gz   # CT (resampled to PET)
  <fdat>/imagesTr/<case>_0001.nii.gz   # PET (SUV)
  <fdat>/labelsTr/<case>.nii.gz        # manual tumor lesion annotation
  <fdat>/splits_final.json             # official 5-fold split
  <fdat>/dataset.json                  # original (will be overwritten with 4ch)

This script:
  1) (optional) binarizes labelsTr to {0,1}
  2) extracts <case>_0002.nii.gz (FG) and <case>_0003.nii.gz (BG) from the
     repo-bundled heatmaps.zip and writes them into imagesTr
  3) writes a 4-channel dataset.json (CT/PET/FG/BG)
  4) ensures splits_final.json sits next to the dataset (it does already)

Usage:
  python scripts/prep_fdat.py \
      --dataset D:/autopet/dataset/PSMA-FDG-PET-CT-Lesions_v2 \
      --heatmaps_zip C:/Users/USER/autoPETV/nnunet-baseline/heatmaps.zip \
      [--skip_binarize]
"""
import os
import sys
import json
import zipfile
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import nibabel as nib


def list_cases(labels_dir: str) -> list[str]:
    return sorted(f[:-7] for f in os.listdir(labels_dir) if f.endswith(".nii.gz"))


def binarize_label(p: str) -> None:
    im = nib.load(p)
    arr = np.asanyarray(im.dataobj)
    if arr.dtype == np.uint8 and arr.max() <= 1:
        return  # already binary 0/1
    out = (arr > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(out, im.affine, im.header), p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="FDAT dataset root (has imagesTr/, labelsTr/, splits_final.json)")
    ap.add_argument("--heatmaps_zip", required=True,
                    help="repo-bundled heatmaps.zip (1611x2 entries)")
    ap.add_argument("--skip_binarize", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ds = args.dataset
    imagesTr = os.path.join(ds, "imagesTr")
    labelsTr = os.path.join(ds, "labelsTr")
    if not (os.path.isdir(imagesTr) and os.path.isdir(labelsTr)):
        sys.exit(f"imagesTr/labelsTr not found under {ds}")

    cases = list_cases(labelsTr)
    print(f"cases: {len(cases)}")

    # 1) binarize labels (parallel)
    if not args.skip_binarize:
        print("binarizing labels ...")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(binarize_label, os.path.join(labelsTr, c + ".nii.gz"))
                    for c in cases]
            for i, f in enumerate(as_completed(futs), 1):
                _ = f.result()
                if i % 200 == 0:
                    print(f"  binarized {i}/{len(cases)}")

    # 2) extract heatmaps (only matching cases)
    print("indexing heatmaps.zip ...")
    with zipfile.ZipFile(args.heatmaps_zip) as z:
        names = z.namelist()
        # entries look like: heatmaps/<case>_0002.nii.gz  (or just <case>_0002.nii.gz)
        index: dict[str, str] = {}
        for n in names:
            base = os.path.basename(n)
            if base.endswith("_0002.nii.gz") or base.endswith("_0003.nii.gz"):
                index[base] = n
        print(f"heatmap entries indexed: {len(index)}")

        wrote = miss = 0
        for c in cases:
            for ch in ("0002", "0003"):
                key = f"{c}_{ch}.nii.gz"
                src = index.get(key)
                if src is None:
                    miss += 1
                    if miss <= 5:
                        print(f"[WARN] missing in zip: {key}")
                    continue
                with z.open(src) as r, open(os.path.join(imagesTr, key), "wb") as w:
                    w.write(r.read())
                wrote += 1
                if wrote % 400 == 0:
                    print(f"  extracted {wrote}/{len(cases)*2}")
        print(f"extracted heatmaps: {wrote}  missing: {miss}")

    # 3) verify all 4 channels exist per case
    ok = 0
    bad: list[str] = []
    for c in cases:
        if all(os.path.exists(os.path.join(imagesTr, f"{c}_{ch}.nii.gz"))
               for ch in ("0000", "0001", "0002", "0003")):
            ok += 1
        else:
            bad.append(c)
    print(f"complete 4ch cases: {ok}/{len(cases)}")
    if bad:
        with open(os.path.join(ds, "incomplete_cases.txt"), "w") as f:
            f.write("\n".join(bad))
        print(f"incomplete cases listed at: {os.path.join(ds, 'incomplete_cases.txt')}")

    # 4) write 4ch dataset.json (preserve numTraining = ok)
    dataset_json = {
        "channel_names": {"0": "CT", "1": "PET", "2": "FG", "3": "BG"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": ok,
        "file_ending": ".nii.gz",
        "name": "AutoPETV_FDAT_1611",
        "description": "PSMA-FDG-PET-CT-Lesions_v2 + repo-bundled scribble heatmaps",
    }
    with open(os.path.join(ds, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"wrote dataset.json (numTraining={ok})")
    print("splits_final.json present:",
          os.path.exists(os.path.join(ds, "splits_final.json")))


if __name__ == "__main__":
    main()
