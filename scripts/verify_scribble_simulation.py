"""Reproduce the shipped scribble heatmaps from the labels, to pin the recipe.

Why this exists
---------------
The fine-tune has to generate interaction signal on the fly, and it has to
generate the *same kind* of signal the evaluation feeds. We know the shape of
that signal from interactive/simulate_scribbles.py: contiguous strokes on one
2D slice of a component, not scattered points. What we do not know is which
`--strategy` the organizers used to build the shipped heatmaps, or whether our
reading of the seeding is right.

We can settle both without guessing, because nnunet-baseline/heatmaps.zip *is*
the output of that simulator run on the training labels. Running the same
functions on the same labels must reproduce it exactly, the same way the click
coordinates reproduced it byte for byte in the earlier check.

Getting this wrong is the same class of mistake as the dead click channels:
training on a signal distribution that inference never produces, with nothing in
the logs to indicate a problem.

Usage
-----
    python scripts/verify_scribble_simulation.py --n 10
    python scripts/verify_scribble_simulation.py --n 10 --repo D:\\autopet\\official_wt
"""
import argparse
import gzip
import io
import os
import sys
import zipfile
from os.path import isdir, isfile, join

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=r"D:\autopet\official_wt",
                   help="pristine checkout that holds interactive/ and nnunet-baseline/heatmaps.zip")
    p.add_argument("--labels", default=r"D:\autopet\dataset\Dataset998_AutoPETV\labelsTr")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_official(zf, name):
    import nibabel as nib
    return np.asanyarray(nib.Nifti1Image.from_bytes(gzip.decompress(zf.read(name))).dataobj)


def main():
    args = parse_args()

    sim_dir = join(args.repo, "interactive")
    hm_zip = join(args.repo, "nnunet-baseline", "heatmaps.zip")
    for p in (sim_dir, args.labels):
        if not isdir(p):
            sys.exit(f"[FATAL] 見つからない: {p}")
    if not isfile(hm_zip):
        sys.exit(f"[FATAL] 見つからない: {hm_zip}")

    sys.path.insert(0, sim_dir)
    try:
        from simulate_scribbles import (get_random_k_components,
                                        generate_scribbles_for_components)
    except Exception as e:
        sys.exit(f"[FATAL] simulate_scribbles を import できない: {e}\n"
                 "  依存: cc3d(connected-components-3d) / networkx / scikit-image / scipy\n"
                 "  例: C:\\miniconda\\envs\\autopetv\\python.exe -m pip install "
                 "connected-components-3d networkx scikit-image")
    import nibabel as nib  # noqa: F401  (load_official needs it)

    zf = zipfile.ZipFile(hm_zip)
    names = {n.split("/")[-1]: n for n in zf.namelist() if n.endswith(".nii.gz")}
    tags = sorted({n.rsplit("_", 1)[0] for n in names})

    # only cases whose official FG heatmap is non-empty are informative
    picked, i = [], 0
    while len(picked) < args.n and i < len(tags):
        tag = tags[i]
        i += 1
        fn = f"{tag}_0002.nii.gz"
        if fn not in names or not isfile(join(args.labels, f"{tag}.nii.gz")):
            continue
        arr = load_official(zf, names[fn])
        if np.count_nonzero(arr):
            picked.append((tag, {tuple(c) for c in np.argwhere(arr > 0)}))
        del arr

    print(f"検証対象 {len(picked)} 症例\n")
    strategies = ("centerline", "boundary", "random")
    score = {s: 0 for s in strategies}

    for tag, official in picked:
        label = np.asanyarray(nib.load(join(args.labels, f"{tag}.nii.gz")).dataobj).astype(np.uint8)
        line = f"{tag[:44]:44s} 公式={len(official):5d}"
        for s in strategies:
            labels_fg, ids = get_random_k_components(label, k=5)
            vol = generate_scribbles_for_components(labels_fg, ids, s, args.seed)
            mine = {tuple(c) for c in np.argwhere(vol > 0)}
            exact = mine == official
            score[s] += exact
            inter = len(mine & official)
            line += f" | {s[:4]}={len(mine):5d} 一致={inter:5d}{' ★完全一致' if exact else ''}"
        print(line)
        del label

    print("\n=== strategy 別の完全一致数 ===")
    for s in strategies:
        print(f"  {s:<11} {score[s]} / {len(picked)}")
    winner = [s for s in strategies if score[s] == len(picked)]
    if winner:
        print(f"\n公式が使った strategy = {winner[0]} で確定。"
              " fine-tune のクリック生成はこれに合わせること。")
    else:
        print("\n⚠️ どの strategy も完全再現しない。seed の与え方かBG領域の作り方が違う。"
              " 一致数の列を見て、部分一致しているものから詰めること。"
              " ここが合わないまま学習を回すと、訓練信号が本番と別物になる。")


if __name__ == "__main__":
    main()
