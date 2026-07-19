"""
Regenerate the FG/BG click channels (_0002 / _0003) for Dataset998, in place.

Background: the original prep_fdat.py read clicks as data["tumor"] and
data["background"], but lesion-clicks.json actually stores
    {"points": [{"point": [x, y, z], "name": "tumor"|"background"}, ...]}
so both lookups returned [] and every case got an all-zero heatmap -- written
without raising, and reported as "errors: 0". All 1611 preprocessed cases have
dead click channels, which is why every model we trained ignores clicks: 50
background clicks move its prediction by exactly 0 voxels, while the official
baseline gains +0.026 Dice from 4.

It also sized the volume with SimpleITK (array order z,y,x) while the clicks are
[x, y, z]. Verified against ground truth (verify_click_axis_order.py): indexing
the nibabel array directly as arr[c0, c1, c2] puts 42/42 tumor clicks inside the
lesion and 0/67 background clicks -- every other permutation scores 0-4/42. So
this script uses nibabel, not SimpleITK.

Only _0002/_0003 are touched; CT, PET and labels are left alone.

Because the original bug was a SILENT one, this script refuses to be quiet: it
counts what it wrote and verifies that regenerated tumor clicks really land on
lesions, and exits non-zero if they do not.

Run on PC-C:
  <autopetv-python> scripts/regen_click_heatmaps.py
"""
import os
import sys
import json
import zipfile
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import nibabel as nib

IMAGES_DEFAULT = r"D:\autopet\dataset\Dataset998_AutoPETV\imagesTr"
LABELS_DEFAULT = r"D:\autopet\dataset\Dataset998_AutoPETV\labelsTr"
ZIP_DEFAULT = r"C:\Users\OkudaLab08\autoPETV\nnunet-baseline\lesion-scribbles.zip"


def parse_points(raw):
    """lesion-clicks.json -> (tumor_coords, background_coords).

    Fails loudly on an unexpected layout: silently returning [] is the exact bug
    this script exists to fix.
    """
    d = json.loads(raw)
    if "points" not in d:
        raise KeyError(f"no 'points' key; got keys={list(d.keys())}")
    fg, bg, unknown = [], [], set()
    for p in d["points"]:
        name = p.get("name")
        coord = p.get("point")
        if coord is None or len(coord) != 3:
            raise ValueError(f"bad point entry: {p}")
        if name == "tumor":
            fg.append(coord)
        elif name == "background":
            bg.append(coord)
        else:
            unknown.add(name)
    if unknown:
        raise ValueError(f"unexpected point name(s): {sorted(unknown)}")
    return fg, bg


def write_heatmap(coords, ref_img, out_path):
    """Write a 0/1 heatmap in the reference image's geometry.

    Coords are [x, y, z] indices into the nibabel array (verified convention).
    Returns (n_written, n_out_of_bounds).
    """
    shape = ref_img.shape
    arr = np.zeros(shape, dtype=np.float32)
    written = oob = 0
    for c in coords:
        if all(0 <= c[i] < shape[i] for i in range(3)):
            arr[c[0], c[1], c[2]] = 1.0
            written += 1
        else:
            oob += 1
    out = nib.Nifti1Image(arr, ref_img.affine, ref_img.header)
    out.set_data_dtype(np.float32)
    nib.save(out, out_path)
    return written, oob


def process(task):
    case, images_dir, raw_json = task
    pet_path = os.path.join(images_dir, f"{case}_0001.nii.gz")
    try:
        fg, bg = parse_points(raw_json)
        pet = nib.load(pet_path)
        nfg, oob_fg = write_heatmap(fg, pet, os.path.join(images_dir, f"{case}_0002.nii.gz"))
        nbg, oob_bg = write_heatmap(bg, pet, os.path.join(images_dir, f"{case}_0003.nii.gz"))
        return case, True, nfg, nbg, oob_fg + oob_bg, None
    except Exception as e:
        return case, False, 0, 0, 0, f"{type(e).__name__}: {e}"


def verify(images_dir, labels_dir, cases, n_check):
    """Regenerated tumor clicks must sit inside the lesion mask."""
    print(f"\n=== verification: do regenerated FG clicks land on lesions? "
          f"({n_check} cases with clicks) ===")
    tot_on = tot = checked = 0
    for case in cases[:n_check]:
        lab_path = os.path.join(labels_dir, f"{case}.nii.gz")
        fg_path = os.path.join(images_dir, f"{case}_0002.nii.gz")
        if not (os.path.exists(lab_path) and os.path.exists(fg_path)):
            continue
        gt = np.asanyarray(nib.load(lab_path).dataobj) > 0
        fg = np.asanyarray(nib.load(fg_path).dataobj) > 0
        n = int(fg.sum())
        if n == 0:
            continue
        on = int((fg & gt).sum())
        tot_on += on
        tot += n
        checked += 1
        print(f"  {case[:52]:<52} {on:>4d}/{n:<4d} on lesion")
    rate = tot_on / tot if tot else 0.0
    print(f"\n  overall: {tot_on}/{tot} FG click voxels inside the lesion "
          f"({rate:.1%}) over {checked} cases")
    return rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=IMAGES_DEFAULT)
    ap.add_argument("--labels", default=LABELS_DEFAULT)
    ap.add_argument("--zip", dest="zip_path", default=ZIP_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify_n", type=int, default=15)
    args = ap.parse_args()

    cases = sorted(f[:-len("_0001.nii.gz")] for f in os.listdir(args.images)
                   if f.endswith("_0001.nii.gz"))
    print(f"cases found in imagesTr : {len(cases)}")

    with zipfile.ZipFile(args.zip_path) as z:
        entries = {n: z.read(n) for n in z.namelist()}
    print(f"entries in scribbles zip: {len(entries)}")

    tasks, missing = [], []
    for c in cases:
        key = f"{c}_lesion-clicks.json"
        if key in entries:
            tasks.append((c, args.images, entries[key]))
        else:
            missing.append(c)
    print(f"matched                 : {len(tasks)}   missing json: {len(missing)}")
    if missing:
        print("  e.g. " + "; ".join(missing[:3]))

    ok = err = 0
    empty_fg = empty_bg = 0
    tot_fg = tot_bg = tot_oob = 0
    failures = []
    with_clicks = []

    print("\nregenerating heatmaps ...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (case, good, nfg, nbg, oob, msg) in enumerate(ex.map(process, tasks, chunksize=8), 1):
            if good:
                ok += 1
                tot_fg += nfg
                tot_bg += nbg
                tot_oob += oob
                empty_fg += (nfg == 0)
                empty_bg += (nbg == 0)
                if nfg:
                    with_clicks.append(case)
            else:
                err += 1
                failures.append((case, msg))
            if i % 200 == 0:
                print(f"  {i}/{len(tasks)}")

    print(f"\n==== WRITE SUMMARY ====")
    print(f"ok                 : {ok}      errors: {err}")
    print(f"FG click voxels    : {tot_fg}   (cases with none: {empty_fg})")
    print(f"BG click voxels    : {tot_bg}   (cases with none: {empty_bg})")
    print(f"out-of-bounds drops: {tot_oob}")
    for case, msg in failures[:5]:
        print(f"  FAIL {case[:50]}: {msg}")

    if tot_fg == 0 and tot_bg == 0:
        print("\n>> ABORT: still wrote nothing. Do NOT preprocess.")
        sys.exit(1)

    rate = verify(args.images, args.labels, with_clicks, args.verify_n)

    print("\n==== VERDICT ====")
    if rate >= 0.9:
        print(f">> Click channels are alive and correctly placed ({rate:.1%} on lesion).")
        print(">> Safe to re-run preprocessing, then retrain.")
    else:
        print(f">> STOP: only {rate:.1%} of FG clicks land on lesions -- geometry is wrong.")
        print(">> Do not preprocess or retrain until this is resolved.")
        sys.exit(1)


if __name__ == "__main__":
    main()
