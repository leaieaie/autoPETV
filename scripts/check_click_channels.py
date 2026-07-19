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
    ap.add_argument("--n", type=int, default=12, help="how many cases to inspect")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.prep)
                   if f.endswith(".b2nd") and not f.endswith("_seg.b2nd"))
    if not files:
        raise SystemExit(f"no .b2nd data files under {args.prep}")
    print(f"found {len(files)} preprocessed cases; inspecting first {min(args.n, len(files))}\n")

    n_ch2_empty = n_ch3_empty = 0
    inspected = 0

    hdr = f"{'case':<42} {'ch2(FG) nonzero':>16} {'ch3(BG) nonzero':>16}  {'ch2 max':>9} {'ch3 max':>9}"
    print(hdr)
    print("-" * len(hdr))

    for f in files[:args.n]:
        data = load_case(os.path.join(args.prep, f))
        if data.shape[0] < 4:
            print(f"{f:<42} !! only {data.shape[0]} channels")
            continue
        ch2, ch3 = data[2], data[3]
        nz2, nz3 = int(np.count_nonzero(ch2)), int(np.count_nonzero(ch3))
        n_ch2_empty += (nz2 == 0)
        n_ch3_empty += (nz3 == 0)
        inspected += 1
        print(f"{f[:42]:<42} {nz2:>16d} {nz3:>16d}  {float(ch2.max()):>9.3f} {float(ch3.max()):>9.3f}")

    print("\n==== VERDICT ====")
    print(f"inspected            : {inspected}")
    print(f"ch2 (FG) all-zero    : {n_ch2_empty}/{inspected}")
    print(f"ch3 (BG) all-zero    : {n_ch3_empty}/{inspected}")
    if inspected and n_ch3_empty == inspected:
        print(">> ch3 (background clicks) is EMPTY in every inspected case.")
        print(">> Confirms the bug: models cannot learn to use BG clicks -> FPs never removed -> low F1/DMM.")
    elif inspected and n_ch3_empty:
        print(">> ch3 is empty in SOME cases -- partial BG-click coverage.")
    else:
        print(">> ch3 carries signal; the BG-blindness must come from elsewhere (training/normalization).")


if __name__ == "__main__":
    main()
