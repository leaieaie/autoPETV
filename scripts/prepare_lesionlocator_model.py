"""Build a _model folder for the LesionLocator container from the released archive.

Why this exists
---------------
The published checkpoint is 8.2 GB: ten ResEncL folds at roughly 820 MB each.
Shipping all ten would blow both the per-iteration time budget (600 s) and the
image size, and most of each file is optimizer state that inference never reads.

This extracts the folds we actually want and rewrites each checkpoint keeping
only the keys nnU-Net's predictor loads, then reloads what it wrote to prove the
result is still usable. Nothing is deleted from the archive.

Usage
-----
    python prepare_lesionlocator_model.py --list
    python prepare_lesionlocator_model.py --folds 0 1
"""
import argparse
import io
import json
import os
import shutil
import zipfile
from os.path import isdir, isfile, join

import torch

# initialize_from_trained_model_folder reads exactly these; everything else in
# the checkpoint (optimizer_state, grad_scaler_state, logging, _best_ema) is
# training bookkeeping.
KEEP = ("network_weights", "trainer_name", "inference_allowed_mirroring_axes", "init_args")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default=r"D:\autopet\lesionlocator_model.zip")
    p.add_argument("--out", default=r"D:\autopet\autoPET-interactive\_model")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--list", action="store_true", help="展開せず中身だけ表示")
    p.add_argument("--keep-optimizer", action="store_true", help="軽量化しない")
    return p.parse_args()


def main():
    args = parse_args()
    if not isfile(args.zip):
        raise SystemExit(f"[FATAL] zip が無い: {args.zip}")
    zf = zipfile.ZipFile(args.zip)
    infos = [i for i in zf.infolist() if not i.is_dir()]
    root = os.path.commonprefix([i.filename for i in infos]).split("/")[0]

    if args.list:
        print(f"root = {root}")
        for i in sorted(infos, key=lambda x: x.filename):
            print(f"  {i.file_size/1e6:9.1f} MB  {i.filename}")
        return

    os.makedirs(args.out, exist_ok=True)

    # root-level metadata: plans.json / dataset.json / dataset_fingerprint.json
    meta = [i for i in infos if i.filename.count("/") == 1]
    if not meta:
        print("[WARN] zip の直下に json が無い。リポジトリ側の _model にある想定か要確認")
    for i in meta:
        name = i.filename.split("/")[-1]
        with zf.open(i) as src, open(join(args.out, name), "wb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"[meta] {name}  {i.file_size/1e6:.2f} MB")

    total_before = total_after = 0
    for f in args.folds:
        src_name = f"{root}/fold_{f}/checkpoint_final.pth"
        if src_name not in zf.namelist():
            print(f"[WARN] {src_name} が無い。飛ばす")
            continue
        dst_dir = join(args.out, f"fold_{f}")
        os.makedirs(dst_dir, exist_ok=True)
        dst = join(dst_dir, "checkpoint_final.pth")

        raw = zf.read(src_name)
        total_before += len(raw)
        if args.keep_optimizer:
            with open(dst, "wb") as fh:
                fh.write(raw)
        else:
            ck = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            missing = [k for k in KEEP if k not in ck]
            if missing:
                raise SystemExit(f"[FATAL] fold_{f}: 期待するキーが無い {missing}. "
                                 "--keep-optimizer で無加工のまま置くこと")
            torch.save({k: ck[k] for k in KEEP}, dst)
            del ck
        size = os.path.getsize(dst)
        total_after += size

        # reload what we just wrote, so a broken strip fails here and not in the container
        back = torch.load(dst, map_location="cpu", weights_only=False)
        n_par = sum(v.numel() for v in back["network_weights"].values())
        print(f"[fold_{f}] {len(raw)/1e6:7.1f} MB -> {size/1e6:7.1f} MB  "
              f"params={n_par/1e6:.1f}M trainer={back.get('trainer_name')} "
              f"mirror={back.get('inference_allowed_mirroring_axes')}")
        del back

    print(f"\n合計 {total_before/1e9:.2f} GB -> {total_after/1e9:.2f} GB  ({args.out})")
    print("残り: inference.py の use_folds と use_mirroring を合わせること")


if __name__ == "__main__":
    main()
