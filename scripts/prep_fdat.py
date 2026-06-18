"""
Assemble the FDAT (PSMA-FDG-PET-CT-Lesions_v2) dataset into nnU-Net 4-channel
format for autoPET V (Dataset998_AutoPETV).

Key fix vs first version: do NOT use the repo-bundled heatmaps.zip (its NIfTIs
have geometries that disagree with FDAT v2 for some cases). Instead, REGENERATE
the FG/BG scribble heatmaps from the repo-bundled lesion-scribbles.zip (click
coordinates in JSON), using each case's own PET image as the geometric
reference. This guarantees per-case geometry alignment (origin/direction/shape).

Inputs are already nnU-Net-ish:
  <fdat>/imagesTr/<case>_0000.nii.gz   # CT (resampled to PET grid)
  <fdat>/imagesTr/<case>_0001.nii.gz   # PET (SUV)
  <fdat>/labelsTr/<case>.nii.gz        # manual tumor lesion annotation
  <fdat>/splits_final.json
  <fdat>/dataset.json                  # original (will be overwritten with 4ch)

This script:
  1) (optional) binarizes labelsTr to {0,1}
  2) regenerates <case>_0002.nii.gz (FG) and <case>_0003.nii.gz (BG) for every
     case, using PET geometry + lesion-clicks.json from lesion-scribbles.zip
  3) writes a 4-channel dataset.json (CT/PET/FG/BG)

Usage (PowerShell):
  python scripts/prep_fdat.py \
      --dataset D:/autopet/dataset/Dataset998_AutoPETV \
      --scribbles_zip C:/Users/USER/autoPETV/nnunet-baseline/lesion-scribbles.zip
"""
import os
import sys
import json
import zipfile
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import SimpleITK as sitk


def binarize_label_inplace(p: str) -> None:
    im = sitk.ReadImage(p)
    arr = sitk.GetArrayFromImage(im)
    if arr.dtype == np.uint8 and arr.max() <= 1:
        return
    out = (arr > 0).astype(np.uint8)
    new = sitk.GetImageFromArray(out)
    new.CopyInformation(im)
    sitk.WriteImage(new, p, useCompression=True)


def write_heatmap(coords_zyx: list, pet_path: str, out_path: str) -> None:
    """coords are (z, y, x) in voxel index space (nibabel/sitk array order)."""
    pet = sitk.ReadImage(pet_path)
    shape = sitk.GetArrayFromImage(pet).shape  # (z, y, x)
    arr = np.zeros(shape, dtype=np.float32)
    for c in coords_zyx:
        if (0 <= c[0] < shape[0]
                and 0 <= c[1] < shape[1]
                and 0 <= c[2] < shape[2]):
            arr[c[0], c[1], c[2]] = 1.0
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(pet)
    sitk.WriteImage(img, out_path, useCompression=True)


def _process_case(args):
    case, imagesTr, clicks_json = args
    pet = os.path.join(imagesTr, f"{case}_0001.nii.gz")
    if not os.path.exists(pet):
        return case, False, "no PET"
    try:
        data = json.loads(clicks_json)
    except Exception as e:
        return case, False, f"bad json: {e}"
    # spec: utils.save_click_heatmaps used coords directly as (z, y, x) array
    # indices written via nibabel; SimpleITK array order is also (z, y, x), so
    # the click list maps 1:1 to numpy index.
    fg = data.get("tumor", []) or []
    bg = data.get("background", []) or []
    try:
        write_heatmap(fg, pet, os.path.join(imagesTr, f"{case}_0002.nii.gz"))
        write_heatmap(bg, pet, os.path.join(imagesTr, f"{case}_0003.nii.gz"))
    except Exception as e:
        return case, False, f"write: {e}"
    return case, True, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scribbles_zip", required=True,
                    help="repo-bundled lesion-scribbles.zip (1611 JSONs)")
    ap.add_argument("--skip_binarize", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ds = args.dataset
    imagesTr = os.path.join(ds, "imagesTr")
    labelsTr = os.path.join(ds, "labelsTr")
    if not (os.path.isdir(imagesTr) and os.path.isdir(labelsTr)):
        sys.exit(f"imagesTr/labelsTr not found under {ds}")

    cases = sorted(f[:-7] for f in os.listdir(labelsTr) if f.endswith(".nii.gz"))
    print(f"cases: {len(cases)}")

    # 1) binarize labels
    if not args.skip_binarize:
        print("binarizing labels ...")
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(binarize_label_inplace,
                              os.path.join(labelsTr, c + ".nii.gz")) for c in cases]
            for i, _ in enumerate(as_completed(futs), 1):
                if i % 300 == 0:
                    print(f"  binarized {i}/{len(cases)}")

    # 2) load click jsons from zip
    print("loading clicks JSONs ...")
    case_to_json: dict[str, str] = {}
    missing: list[str] = []
    with zipfile.ZipFile(args.scribbles_zip) as z:
        names = {os.path.basename(n): n for n in z.namelist() if n.endswith(".json")}
        for c in cases:
            entry = names.get(f"{c}_lesion-clicks.json")
            if entry is None:
                missing.append(c)
                continue
            with z.open(entry) as f:
                case_to_json[c] = f.read().decode("utf-8")
    print(f"clicks found: {len(case_to_json)} / missing: {len(missing)}")
    if missing[:5]:
        print("first missing:", missing[:5])

    # 3) regenerate _0002/_0003 with PET geometry (parallel)
    print("regenerating heatmaps with PET geometry ...")
    todo = [(c, imagesTr, case_to_json[c]) for c in case_to_json]
    done = err = 0
    errs: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed(ex.submit(_process_case, t) for t in todo):
            case, ok, msg = fut.result()
            if ok:
                done += 1
                if done % 200 == 0:
                    print(f"  wrote {done}/{len(todo)} heatmaps")
            else:
                err += 1
                errs.append(f"{case}: {msg}")
    print(f"heatmaps written: {done}  errors: {err}")
    if err:
        with open(os.path.join(ds, "heatmap_errors.txt"), "w") as f:
            f.write("\n".join(errs))
        print(f"errors listed at: {os.path.join(ds, 'heatmap_errors.txt')}")

    # 4) verify all 4 channels exist per case
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

    # 5) write 4ch dataset.json
    dataset_json = {
        "channel_names": {"0": "CT", "1": "PET", "2": "FG", "3": "BG"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": ok,
        "file_ending": ".nii.gz",
        "name": "AutoPETV_FDAT_1611",
        "description": "PSMA-FDG-PET-CT-Lesions_v2; heatmaps regenerated from lesion-scribbles.zip using each PET as reference (geometry-safe).",
    }
    with open(os.path.join(ds, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"wrote dataset.json (numTraining={ok})")
    print("splits_final.json present:",
          os.path.exists(os.path.join(ds, "splits_final.json")))


if __name__ == "__main__":
    main()
