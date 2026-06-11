"""
Assemble a DeepPSMA -> nnU-Net 4-channel dataset for autoPET V (Dataset998 format).

Run AFTER convert_deep_psma_to_nnunet_format.py has produced imagesTr (_0000 CT,
_0001 PET) and labelsTr (<case>.nii.gz). This script then:
  1. binarizes labels (TTB > 0 -> 1) so labels are {0,1}
  2. extracts the matching scribble heatmaps (_0002 FG, _0003 BG) from
     heatmaps_deep_psma.zip into imagesTr (only for cases actually present)
  3. writes dataset.json (4 channels: CT/PET/FG/BG)

Usage:
  python scripts/prep_deeppsma.py \
      --dataset <nnUNet_raw>/Dataset998_AutoPETV \
      --heatmaps_zip DeepPSMA/heatmaps_deep_psma.zip
"""
import os
import sys
import json
import zipfile
import argparse

import numpy as np
import nibabel as nib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset998_AutoPETV dir (has imagesTr, labelsTr)")
    ap.add_argument("--heatmaps_zip", required=True, help="DeepPSMA/heatmaps_deep_psma.zip")
    args = ap.parse_args()

    ds = args.dataset
    imagesTr = os.path.join(ds, "imagesTr")
    labelsTr = os.path.join(ds, "labelsTr")
    if not os.path.isdir(labelsTr):
        sys.exit(f"labelsTr not found under {ds}. Run convert_deep_psma_to_nnunet_format.py first.")

    cases = sorted(f[:-7] for f in os.listdir(labelsTr) if f.endswith(".nii.gz"))
    print(f"Found {len(cases)} cases in labelsTr")

    # 1) binarize labels -> {0,1}
    for c in cases:
        p = os.path.join(labelsTr, c + ".nii.gz")
        im = nib.load(p)
        d = (np.asanyarray(im.dataobj) > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(d, im.affine, im.header), p)

    # 2) extract matching heatmaps into imagesTr
    with zipfile.ZipFile(args.heatmaps_zip) as z:
        names = z.namelist()
        nameset = set(names)
        for c in cases:
            for ch in ("0002", "0003"):
                entry = f"{c}_{ch}.nii.gz"
                match = entry if entry in nameset else next(
                    (n for n in names if n.endswith("/" + entry) or n == entry), None)
                if match is None:
                    print(f"[WARN] heatmap missing in zip: {entry}")
                    continue
                with z.open(match) as src, open(os.path.join(imagesTr, entry), "wb") as dst:
                    dst.write(src.read())

    # 3) verify 4 channels per case
    ok = 0
    for c in cases:
        if all(os.path.exists(os.path.join(imagesTr, f"{c}_{ch}.nii.gz"))
               for ch in ("0000", "0001", "0002", "0003")):
            ok += 1
        else:
            print(f"[WARN] incomplete channels: {c}")
    print(f"{ok}/{len(cases)} cases have all 4 channels (CT/PET/FG/BG)")

    # 4) dataset.json
    dataset_json = {
        "channel_names": {"0": "CT", "1": "PET", "2": "FG", "3": "BG"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": "AutoPETV_DeepPSMA_subset",
    }
    with open(os.path.join(ds, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"Wrote dataset.json (numTraining={len(cases)})")


if __name__ == "__main__":
    main()
