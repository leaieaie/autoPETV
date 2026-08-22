"""Say which trained model each checkpoint actually is, by weights not by path.

Two ResEncM checkpoints exist and they are the same size, the same architecture,
and both carry trainer_name 'nnUNetTrainer' once patched for the container:

    nnUNetTrainer_OnTheFlyClicks__nnUNetResEncUNetMPlans   ResEncM OTF, LB 0.6805/0.7059
    nnUNetTrainer__nnUNetResEncUNetMPlans                  dead-click ResEncM, LB 0.5132

Nothing about the file distinguishes them from outside. The container copy has
already been wrong once for the official baseline, and shipping the dead-click
model inside the ensemble would quietly cost a submission, so match the weights.

Usage
-----
    python scripts/identify_checkpoints.py
"""
import hashlib
import os
from os.path import basename, dirname, isfile

import torch

CONTAINER = r"C:\Users\OkudaLab08\autoPETV\nnunet-baseline\nnUNet_results\Dataset998_AutoPETV"
RESULTS = r"D:\autopet\nnUNet\results\Dataset998_AutoPETV"

CANDIDATES = {
    "コンテナ: nnUNetPlans (公式であるべき)":
        rf"{CONTAINER}\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_final.pth",
    "コンテナ: ResEncMPlans (OTFであるべき)":
        rf"{CONTAINER}\nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres\fold_0\checkpoint_final.pth",
    "学習結果: ResEncM-OTF":
        rf"{RESULTS}\nnUNetTrainer_OnTheFlyClicks__nnUNetResEncUNetMPlans__3d_fullres\fold_0\checkpoint_final.pth",
    "学習結果: ResEncM dead-click (残骸)":
        rf"{RESULTS}\nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres\fold_0\checkpoint_final.pth",
    "学習結果: PlainConv OTF":
        rf"{RESULTS}\nnUNetTrainer_OnTheFlyClicks__nnUNetPlans__3d_fullres\fold_0\checkpoint_final.pth",
    "学習結果: click修正版 標準":
        rf"{RESULTS}\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_final.pth",
}

# from the official training log shipped in the repository
OFFICIAL = (1000, 0.1041, 0.8349, 0.7043)


def describe(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    w = ck["network_weights"]
    key = sorted(w.keys())[0]
    digest = hashlib.md5(w[key].float().numpy().tobytes()).hexdigest()[:12]
    hist = (ck.get("logging") or {}).get("mean_fg_dice") or []
    stats = (len(hist),
             round(float(hist[0]), 4) if hist else None,
             round(float(max(hist)), 4) if hist else None,
             round(float(hist[-1]), 4) if hist else None)
    return digest, ck.get("trainer_name"), stats, os.path.getsize(path) / 1e6


def main():
    seen = {}
    rows = []
    for name, path in CANDIDATES.items():
        if not isfile(path):
            print(f"{name:<40} 見つからない: {path}")
            continue
        digest, trainer, stats, mb = describe(path)
        seen.setdefault(digest, []).append(name)
        rows.append((name, digest, trainer, stats, mb))
        official = "  ← 公式ログと一致" if stats == OFFICIAL else ""
        print(f"{name:<40} {mb:6.0f} MB  hash={digest}  trainer={trainer}")
        print(f"{'':<40} epochs={stats[0]} first={stats[1]} best={stats[2]} last={stats[3]}{official}")

    print("\n=== 同一の重み ===")
    for digest, names in seen.items():
        mark = "  ⚠️ 同一" if len(names) > 1 else ""
        print(f"  {digest}: {', '.join(names)}{mark}")

    def find(label):
        return next((r for r in rows if r[0].startswith(label)), None)

    cont = find("コンテナ: ResEncMPlans")
    otf = find("学習結果: ResEncM-OTF")
    dead = find("学習結果: ResEncM dead-click")
    print("\n=== 判定 ===")
    if cont is None:
        print("  コンテナの ResEncM が無い")
    elif otf and cont[1] == otf[1]:
        print("  ✅ コンテナの ResEncM は OTF 版。ビルドしてよい")
    elif dead and cont[1] == dead[1]:
        print("  ❌ コンテナの ResEncM は dead-click 版(LB 0.5132)。"
              " OTF 版で上書きしてからビルドすること")
    else:
        print("  ⚠️ どちらとも一致しない。上のハッシュを見て確認すること")


if __name__ == "__main__":
    main()
