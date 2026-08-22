"""Diff everything about our dataset against the organizers' own, except the voxels.

Why this exists
---------------
We verified the click channels byte for byte against heatmaps.zip, but never
checked CT, PET, the labels, the case list, or the plans that were in force when
the data was preprocessed. prep_fdat.py already shipped one silent bug and
fix_labels_geometry.py exists because the labels were wrong at some point, so
"our images are the same as theirs" is an assumption, not a finding.

Four things are compared, all from files that already exist:

  1. dataset_fingerprint.json  per-channel foreground intensities, shapes, spacings
  2. case identifiers          ours (splits_final.json) against theirs (heatmaps.zip)
  3. dataset.json              channel order, labels, numTraining
  4. plans.json                patch size, spacing, normalisation, and the intensity
                               constants that CTNormalization bakes into the data

Point 4 matters on its own: the plans file in our preprocessed folder was
restored from the fork after going missing, so the constants used at preprocessing
time may not be the ones training assumed. Normalisation is baked in at
preprocessing time and cannot be seen from the checkpoint afterwards.

A difference here is a live explanation for the 0.041 validation Dice gap between
the official baseline and our best retrain, and worth more than any architecture
work.

Usage
-----
    python scripts/compare_fingerprints.py
"""
import argparse
import json
import zipfile
from os.path import isfile, join

KEYS = ("mean", "median", "std", "min", "max", "percentile_00_5", "percentile_99_5")
CHANNEL_NAMES = {"0": "CT", "1": "PET", "2": "FG clicks", "3": "BG clicks"}
PLANS_FIELDS = ("data_identifier", "patch_size", "spacing", "batch_size",
                "normalization_schemes", "use_mask_for_norm", "median_image_size_in_voxels")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preprocessed", default=r"D:\autopet\nnUNet\preprocessed\Dataset998_AutoPETV")
    p.add_argument("--official", default=r"D:\autopet\official_wt\nnunet-baseline\nnUNet_results"
                                        r"\Dataset998_AutoPETV\nnUNetTrainer__nnUNetPlans__3d_fullres")
    p.add_argument("--heatmaps", default=r"D:\autopet\official_wt\nnunet-baseline\heatmaps.zip")
    p.add_argument("--tol", type=float, default=0.01, help="相対差の許容値 (既定 1%)")
    return p.parse_args()


def rel(a, b):
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def load(path, label):
    if not isfile(path):
        print(f"  [skip] {label} が無い: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cmp_intensities(ours, off, tol):
    bad = 0
    o_ch = (ours or {}).get("foreground_intensity_properties_per_channel", {})
    f_ch = (off or {}).get("foreground_intensity_properties_per_channel", {})
    for ch in sorted(set(o_ch) | set(f_ch), key=lambda x: int(x)):
        name = CHANNEL_NAMES.get(ch, f"ch{ch}")
        a, b = o_ch.get(ch), f_ch.get(ch)
        if a is None or b is None:
            print(f"  [{name}] 片方に存在しない")
            bad += 1
            continue
        lines = []
        for k in KEYS:
            if k not in a or k not in b:
                continue
            r = rel(float(a[k]), float(b[k]))
            if r > tol:
                bad += 1
                lines.append(f"    {k:<16} ours={float(a[k]):14.6f}  official={float(b[k]):14.6f}"
                             f"  相対差={r:.2%}  <-- 差分")
        print(f"  [{name}] {'一致' if not lines else str(len(lines)) + ' 項目で差分'}")
        for line in lines:
            print(line)
    return bad


def main():
    args = parse_args()
    bad = 0

    print("=== 1. dataset_fingerprint.json (前景voxelの強度統計) ===")
    ours_fp = load(join(args.preprocessed, "dataset_fingerprint.json"), "ours fingerprint")
    off_fp = load(join(args.official, "dataset_fingerprint.json"), "official fingerprint")
    if ours_fp and off_fp:
        bad += cmp_intensities(ours_fp, off_fp, args.tol)
        for key in ("shapes_after_crop", "spacings"):
            a, b = ours_fp.get(key), off_fp.get(key)
            if not (isinstance(a, list) and isinstance(b, list)):
                continue
            if len(a) != len(b):
                print(f"  {key}: 症例数が違う ours={len(a)} official={len(b)}  <-- 差分")
                bad += 1
                continue
            sa, sb = sorted(map(tuple, a)), sorted(map(tuple, b))
            diff = sum(1 for x, y in zip(sa, sb) if x != y)
            print(f"  {key}: n={len(a)} 不一致={diff}" + ("  <-- 差分" if diff else ""))
            if diff:
                bad += 1
                for x, y in list(zip(sa, sb)):
                    if x != y:
                        print(f"    ours={x}  official={y}")
                        break

    print("\n=== 2. 症例名リスト ===")
    splits = load(join(args.preprocessed, "splits_final.json"), "splits_final")
    if splits and isfile(args.heatmaps):
        ours_names = set(splits[0]["train"]) | set(splits[0]["val"])
        zf = zipfile.ZipFile(args.heatmaps)
        off_names = {n.split("/")[-1].rsplit("_", 1)[0]
                     for n in zf.namelist() if n.endswith(".nii.gz")}
        only_ours = ours_names - off_names
        only_off = off_names - ours_names
        print(f"  ours n={len(ours_names)} / official n={len(off_names)}")
        print(f"  ours にしかない: {len(only_ours)} / official にしかない: {len(only_off)}")
        if only_ours or only_off:
            bad += 1
            for n in list(only_ours)[:3]:
                print(f"    ours only : {n}")
            for n in list(only_off)[:3]:
                print(f"    official  : {n}")
        else:
            print("  ✅ 症例名が完全一致 → splits_final.json の中身も公式と同じ")

    print("\n=== 3. dataset.json ===")
    ours_ds = load(join(args.preprocessed, "dataset.json"), "ours dataset.json")
    off_ds = load(join(args.official, "dataset.json"), "official dataset.json")
    if ours_ds and off_ds:
        for k in ("channel_names", "labels", "numTraining", "file_ending"):
            a, b = ours_ds.get(k), off_ds.get(k)
            same = a == b
            print(f"  {k}: {'一致' if same else 'ours=' + str(a) + '  official=' + str(b) + '  <-- 差分'}")
            if not same:
                bad += 1

    print("\n=== 4. plans.json ===")
    ours_pl = load(join(args.preprocessed, "nnUNetPlans.json"), "ours plans")
    off_pl = load(join(args.official, "plans.json"), "official plans")
    if ours_pl and off_pl:
        oc = ours_pl.get("configurations", {}).get("3d_fullres", {})
        fc = off_pl.get("configurations", {}).get("3d_fullres", {})
        for k in PLANS_FIELDS:
            a, b = oc.get(k), fc.get(k)
            same = a == b
            print(f"  {k}: {'一致' if same else 'ours=' + str(a) + '  official=' + str(b) + '  <-- 差分'}")
            if not same:
                bad += 1
        print("  -- CTNormalization が使う強度定数 --")
        bad += cmp_intensities(ours_pl, off_pl, args.tol)

    print("\n=== 判定 ===")
    if bad == 0:
        print("  ✅ 全項目一致。CT/PET/ラベル/症例名/plans は公式と同じ。"
              " 0.041 差の原因はデータでも前処理設定でもない")
    else:
        print(f"  ❌ 差分 {bad} 件。**0.041 差の候補**。"
              " 該当項目から prep_fdat.py / fix_labels_geometry.py / plans 復元の経緯を疑うこと")


if __name__ == "__main__":
    main()
