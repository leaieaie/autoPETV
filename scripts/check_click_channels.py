"""
Check whether the click channels actually carry signal in the PREPROCESSED
training data.

Motivation: a standard nnUNetTrainer retrained on our Dataset998 segments better
than the official baseline (Dice 0.920 vs 0.870) but is completely unresponsive
to background clicks -- 50 BG clicks changed the prediction by exactly 0 voxels.
The official baseline, by contrast, gains +0.026 Dice from 4 BG clicks alone.
If ch3 (background clicks) is empty across the training set, every model we
train is necessarily blind to BG clicks, which is exactly the mechanism for
removing false positives -> would explain our persistent F1/DMM gap.

Channel layout: 0=CT, 1=PET, 2=foreground(tumor) clicks, 3=background clicks.

Reports, per case, the nonzero voxel count of ch2/ch3 (a click heatmap should
have a small but NONZERO count). A ch3 that is all-zero everywhere is the bug.

Run on PC-C:
  <autopetv-python> scripts/check_click_channels.py
"""
import os
import argparse

import numpy as np

PREP_DEFAULT = r"D:\autopet\nnUNet\preprocessed\Dataset998_AutoPETV\nnUNetPlans_3d_fullres"


def load_case(path):
    """Preprocessed nnUNet data is blosc2 (.b2nd). Returns array [C, Z, Y, X]."""
    import blosc2
    arr = blosc2.open(path, mode="r")
    return arr[:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=PREP_DEFAULT)
    ap.add_argument("--n", type=int, default=40, help="how many cases to inspect")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", action="store_true",
                    help="inspect the alphabetically first N instead of a random sample")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.prep)
                   if f.endswith(".b2nd") and not f.endswith("_seg.b2nd"))
    if not files:
        raise SystemExit(f"no .b2nd data files under {args.prep}")

    # The alphabetically-first cases are all one tracer; sample across the whole
    # set so a per-tracer difference cannot masquerade as a global result.
    if args.head:
        picked = files[:args.n]
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(files), size=min(args.n, len(files)), replace=False)
        picked = [files[i] for i in sorted(idx)]

    from collections import Counter
    total_by_pfx = Counter(f.split("_")[0] for f in files)
    print(f"found {len(files)} preprocessed cases {dict(total_by_pfx)}; "
          f"inspecting {len(picked)} ({'head' if args.head else f'random seed={args.seed}'})\n")

    stats = {}  # prefix -> [n, ch2_empty, ch3_empty]
    hdr = f"{'case':<44} {'ch2(FG) nonzero':>16} {'ch3(BG) nonzero':>16}  {'ch2 max':>9} {'ch3 max':>9}"
    print(hdr)
    print("-" * len(hdr))

    for f in picked:
        data = load_case(os.path.join(args.prep, f))
        if data.shape[0] < 4:
            print(f"{f[:44]:<44} !! only {data.shape[0]} channels")
            continue
        ch2, ch3 = data[2], data[3]
        nz2, nz3 = int(np.count_nonzero(ch2)), int(np.count_nonzero(ch3))
        pfx = f.split("_")[0]
        s = stats.setdefault(pfx, [0, 0, 0])
        s[0] += 1
        s[1] += (nz2 == 0)
        s[2] += (nz3 == 0)
        print(f"{f[:44]:<44} {nz2:>16d} {nz3:>16d}  {float(ch2.max()):>9.3f} {float(ch3.max()):>9.3f}")

    print("\n==== VERDICT (per tracer) ====")
    tot = [0, 0, 0]
    for pfx, (n, e2, e3) in sorted(stats.items()):
        print(f"  {pfx:<6} n={n:<4} ch2(FG) all-zero={e2}/{n}   ch3(BG) all-zero={e3}/{n}")
        tot = [tot[i] + [n, e2, e3][i] for i in range(3)]
    n, e2, e3 = tot
    print(f"  {'TOTAL':<6} n={n:<4} ch2(FG) all-zero={e2}/{n}   ch3(BG) all-zero={e3}/{n}")

    print("\n==== READING ====")
    if n and e2 == n and e3 == n:
        print(">> BOTH click channels are EMPTY in every inspected case.")
        print(">> Models saw only dead click channels -> click-blind by construction.")
        print(">> NOTE: plans fingerprint reports ch2 max=1.0 from the RAW data, so the")
        print("   clicks existed before preprocessing and were lost by it (or never written).")
    elif n and (e2 < n or e3 < n):
        print(">> Click signal IS present in some cases -- coverage is partial, not universal.")
        print(">> Compare the per-tracer rows above: a tracer-specific gap points at prep_fdat.py.")
    else:
        print(">> Click channels carry signal; BG-blindness comes from training, not the data.")


if __name__ == "__main__":
    main()
