"""Diff our dataset fingerprint against the organizers' own.

Why this exists
---------------
We verified the click channels byte for byte against heatmaps.zip, but never
checked CT, PET or the labels. prep_fdat.py already shipped one silent bug, and
fix_labels_geometry.py exists because the labels were wrong at some point, so
"our images are the same as theirs" is an assumption, not a finding.

nnU-Net writes dataset_fingerprint.json during preprocessing: per-channel
foreground intensity statistics, per-case shapes after cropping, and spacings.
The organizers' fingerprint ships in the repository. Comparing the two files
tests the whole assembly, at the level the network actually sees, for free.

If the intensities differ, that is a live explanation for the 0.041 validation
Dice gap between the official baseline and our best retrain, and fixing it is
worth more than any amount of architecture work.

Usage
-----
    python scripts/compare_fingerprints.py
"""
import argparse
import json
from os.path import isfile

KEYS = ("mean", "median", "std", "min", "max", "percentile_00_5", "percentile_99_5")
CHANNEL_NAMES = {"0": "CT", "1": "PET", "2": "FG clicks", "3": "BG clicks"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ours", default=r"D:\autopet\nnUNet\preprocessed\Dataset998_AutoPETV\dataset_fingerprint.json")
    p.add_argument("--official", default=r"D:\autopet\official_wt\nnunet-baseline\nnUNet_results"
                                        r"\Dataset998_AutoPETV\nnUNetTrainer__nnUNetPlans__3d_fullres"
                                        r"\dataset_fingerprint.json")
    p.add_argument("--tol", type=float, default=0.01, help="相対差の許容値 (既定 1%)")
    return p.parse_args()


def rel(a, b):
    if a == b:
        return 0.0
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def main():
    args = parse_args()
    for p in (args.ours, args.official):
        if not isfile(p):
            raise SystemExit(f"[FATAL] 見つからない: {p}")
    ours = json.load(open(args.ours))
    off = json.load(open(args.official))

    print(f"ours     : {args.ours}")
    print(f"official : {args.official}\n")

    bad = 0
    print("=== 前景voxelのチャネル別強度統計 ===")
    o_ch = ours.get("foreground_intensity_properties_per_channel", {})
    f_ch = off.get("foreground_intensity_properties_per_channel", {})
    for ch in sorted(set(o_ch) | set(f_ch), key=lambda x: int(x)):
        name = CHANNEL_NAMES.get(ch, f"ch{ch}")
        a, b = o_ch.get(ch), f_ch.get(ch)
        if a is None or b is None:
            print(f"  [{name}] 片方に存在しない (ours={a is not None} official={b is not None})")
            bad += 1
            continue
        print(f"  [{name}]")
        for k in KEYS:
            if k not in a or k not in b:
                continue
            r = rel(float(a[k]), float(b[k]))
            flag = "  <-- 差分" if r > args.tol else ""
            if r > args.tol:
                bad += 1
            print(f"    {k:<16} ours={float(a[k]):14.6f}  official={float(b[k]):14.6f}  相対差={r:.2%}{flag}")

    print("\n=== 症例数・spacing・shape ===")
    for key in ("shapes_after_crop", "spacings", "spacings_after_resampling",
                "median_relative_size_after_cropping"):
        a, b = ours.get(key), off.get(key)
        if a is None and b is None:
            continue
        if isinstance(a, list) and isinstance(b, list):
            same_n = len(a) == len(b)
            print(f"  {key}: ours n={len(a)} / official n={len(b)}"
                  f"{'' if same_n else '   <-- 症例数が違う'}")
            if not same_n:
                bad += 1
                continue
            # element-wise comparison on sorted values so case ordering does not matter
            sa, sb = sorted(map(tuple, a)), sorted(map(tuple, b))
            diff = sum(1 for x, y in zip(sa, sb) if x != y)
            print(f"    完全一致しない要素: {diff} / {len(sa)}"
                  f"{'' if diff == 0 else '   <-- 差分'}")
            if diff:
                bad += 1
                for x, y in list(zip(sa, sb))[:5]:
                    if x != y:
                        print(f"      ours={x}  official={y}")
        else:
            r = rel(float(a), float(b)) if isinstance(a, (int, float)) else None
            flag = "   <-- 差分" if r is not None and r > args.tol else ""
            print(f"  {key}: ours={a} official={b}{flag}")
            if r is not None and r > args.tol:
                bad += 1

    print("\n=== 判定 ===")
    if bad == 0:
        print("  ✅ 一致。CT/PET/ラベルの組み立ては公式と同じ。0.041差の原因はデータではない")
    else:
        print(f"  ❌ 差分 {bad} 件。**これが 0.041 の差の候補**。"
              " 上の該当項目を見て prep_fdat.py / fix_labels_geometry.py の該当処理を疑うこと")


if __name__ == "__main__":
    main()
