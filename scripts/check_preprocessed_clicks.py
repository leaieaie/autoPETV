"""
Final gate before retraining: did the click spikes survive preprocessing?

The regenerated click channels are single voxels set to 1.0. Preprocessing
resamples to the plan spacing with cubic interpolation and then ZScore-normalizes
each channel, so a sparse spike can be smeared or attenuated. If the clicks did
not survive in a recoverable form, the model would again learn to ignore them --
the exact failure we just spent four training runs chasing.

After ZScore on a near-empty channel the background becomes a small negative
constant and each click becomes a large positive outlier, so we look for a small
number of strongly positive voxels and compare that count with the number of
clicks in the source JSON.

Run on PC-C:
  <autopetv-python> scripts/check_preprocessed_clicks.py
"""
import os
import json
import zipfile
import argparse

import numpy as np

PREP_DEFAULT = r"D:\autopet\nnUNet\preprocessed\Dataset998_AutoPETV\nnUNetPlans_3d_fullres"
ZIP_DEFAULT = r"C:\Users\OkudaLab08\autoPETV\nnunet-baseline\lesion-scribbles.zip"


def load_case(path):
    import blosc2
    return blosc2.open(path, mode="r")[:]


def source_counts(entries, case):
    key = f"{case}_lesion-clicks.json"
    if key not in entries:
        return None, None
    pts = json.loads(entries[key]).get("points", [])
    fg = sum(1 for p in pts if p.get("name") == "tumor")
    bg = sum(1 for p in pts if p.get("name") == "background")
    return fg, bg


def spike_count(ch):
    """Voxels far above the channel's baseline -- i.e. surviving clicks."""
    if float(ch.max()) <= float(ch.min()):
        return 0, 0.0
    thr = ch.min() + 0.5 * (ch.max() - ch.min())
    return int((ch > thr).sum()), float(ch.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=PREP_DEFAULT)
    ap.add_argument("--zip", dest="zip_path", default=ZIP_DEFAULT)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    with zipfile.ZipFile(args.zip_path) as z:
        entries = {n: z.read(n) for n in z.namelist()}

    files = sorted(f for f in os.listdir(args.prep)
                   if f.endswith(".b2nd") and not f.endswith("_seg.b2nd"))

    print(f"{'case':<44} {'FG src':>7} {'FG kept':>8} {'BG src':>7} {'BG kept':>8} {'ch2 max':>9}")
    print("-" * 88)

    checked = 0
    ratios = []
    for f in files:
        if checked >= args.n:
            break
        case = f[:-len(".b2nd")]
        s_fg, s_bg = source_counts(entries, case)
        if not s_fg:            # only inspect cases that actually have clicks
            continue
        data = load_case(os.path.join(args.prep, f))
        k_fg, mx = spike_count(data[2])
        k_bg, _ = spike_count(data[3])
        ratios.append(k_fg / s_fg)
        checked += 1
        print(f"{case[:44]:<44} {s_fg:>7d} {k_fg:>8d} {s_bg:>7d} {k_bg:>8d} {mx:>9.1f}")

    print("\n==== VERDICT ====")
    if not ratios:
        print(">> no cases with clicks inspected -- widen --n")
        return
    med = float(np.median(ratios))
    print(f"cases inspected            : {checked}")
    print(f"median kept/source (FG)    : {med:.2f}")
    if med >= 0.5:
        print(">> Clicks survived preprocessing as distinct spikes. Safe to retrain.")
    elif med > 0:
        print(">> Clicks are attenuated but present; training may still work, though a")
        print("   denser encoding (dilation/Gaussian) would carry the signal better.")
    else:
        print(">> Clicks did NOT survive preprocessing. Do not retrain -- fix the encoding.")


if __name__ == "__main__":
    main()
