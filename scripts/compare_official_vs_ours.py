"""Score the official baseline on our fold-0 validation set, with our aggregation.

Why this exists
---------------
Our click-fixed standard run reports Mean Validation Dice 0.7733 and the
official baseline's training log ends at 0.7225, but the two numbers are not
comparable. nnU-Net drops a case from the mean when tp+fp+fn == 0, and our
summary.json shows what that does here:

    val cases            323
    lesion-free          116
      of which dropped   116   (n_pred == 0 on every one)
    entered the mean     207

So 0.7733 is an average over 207 lesion-positive cases. The official number is
an average over an unknown number of cases: any lesion-free case where the
baseline emits a single false-positive voxel counts as Dice 0.0 instead of
being dropped. Solving 0.7225 * (207 + k) / 207 for the lesion-positive-only
mean shows the ordering flips at k = 15 of 116, and k <= 79 algebraically. The
comparison cannot decide anything as it stands.

This script removes the unknown by scoring the official weights on the same 323
cases and putting both models through one evaluator.

What it does
------------
1. reads splits_final.json, takes fold 0's validation list
2. predicts those cases with the official baseline checkpoint
3. evaluates the official predictions against labelsTr
4. re-evaluates our existing fold_0/validation predictions with the same call,
   so any difference in aggregation is removed rather than assumed away
5. prints both means together with how many cases entered each

Only step 2 needs the GPU. Step 4 reuses predictions that already exist.

Usage
-----
    python scripts/compare_official_vs_ours.py --dry-run     # check paths first
    python scripts/compare_official_vs_ours.py

Add --also-ours to re-predict our model from raw as well. The default reuses
the stored validation predictions, which came from preprocessed data instead of
raw; both paths resample to the same target spacing, so the residual difference
is small, but --also-ours removes it at the cost of a second prediction pass.
"""
import argparse
import json
import math
import os
import sys
from os.path import isdir, isfile, join

DEFAULTS = dict(
    dataset=r"D:\autopet\dataset\Dataset998_AutoPETV",
    preprocessed=r"D:\autopet\nnUNet\preprocessed\Dataset998_AutoPETV",
    results=r"D:\autopet\nnUNet\results\Dataset998_AutoPETV",
    official=r"C:\Users\OkudaLab08\autoPETV\nnunet-baseline\nnUNet_results"
             r"\Dataset998_AutoPETV\nnUNetTrainer__nnUNetPlans__3d_fullres",
    ours_dirname="nnUNetTrainer__nnUNetPlans__3d_fullres",
    out=r"D:\autopet\compare_official",
)


def parse_args():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_', '-')}", default=v)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--also-ours", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def val_cases(preprocessed, fold):
    sf = join(preprocessed, "splits_final.json")
    if not isfile(sf):
        sys.exit(f"[FATAL] splits_final.json が無い: {sf}")
    splits = json.load(open(sf))
    cases = list(splits[fold]["val"])
    print(f"[split] fold {fold}: train {len(splits[fold]['train'])} / val {len(cases)}")
    if len(cases) != 323:
        print(f"[WARN] val が 323 でなく {len(cases)}。公式ログは 1288/323。split がずれている可能性")
    return cases


def channel_files(dataset, case):
    return [join(dataset, "imagesTr", f"{case}_{c}.nii.gz") for c in ("0000", "0001", "0002", "0003")]


def check(args, cases):
    """Fail loudly before spending hours on the GPU."""
    ok = True
    miss = [f for c in cases[:20] for f in channel_files(args.dataset, c) if not isfile(f)]
    if miss:
        ok = False
        print(f"[FATAL] 入力チャネルが見つからない (先頭20症例で {len(miss)} 件):")
        for m in miss[:5]:
            print("   ", m)

    labels = join(args.dataset, "labelsTr")
    if not isdir(labels):
        ok = False
        print(f"[FATAL] labelsTr が無い: {labels}")

    ckpt = join(args.official, f"fold_{args.fold}", "checkpoint_final.pth")
    for f in (join(args.official, "dataset.json"), join(args.official, "plans.json"), ckpt):
        if not isfile(f):
            ok = False
            print(f"[FATAL] 公式モデルのファイルが無い: {f}")
    if isfile(ckpt) and os.path.getsize(ckpt) < 10_000_000:
        ok = False
        print(f"[FATAL] {ckpt} が {os.path.getsize(ckpt)} バイトしかない。"
              " git-lfs のポインタのままの可能性 → リポジトリで 'git lfs pull' を実行")

    ours_val = join(args.results, args.ours_dirname, f"fold_{args.fold}", "validation")
    n_ours = len([f for f in os.listdir(ours_val) if f.endswith(".nii.gz")]) if isdir(ours_val) else 0
    if n_ours == 0:
        ok = False
        print(f"[FATAL] 自前の検証予測が無い: {ours_val}")
    else:
        print(f"[ok] 自前の検証予測 {n_ours} 件: {ours_val}")

    try:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: F401
        from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2  # noqa: F401
        print(f"[ok] torch {torch.__version__} / cuda={torch.cuda.is_available()}")
    except Exception as e:
        ok = False
        print(f"[FATAL] import 失敗: {e}")

    print("[check] OK" if ok else "[check] 失敗。上の [FATAL] を潰してから再実行")
    return ok


def predict(model_dir, fold, cases, dataset, out_dir, device):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    os.makedirs(out_dir, exist_ok=True)
    done = {f[:-7] for f in os.listdir(out_dir) if f.endswith(".nii.gz")}
    todo = [c for c in cases if c not in done]
    print(f"[predict] {model_dir}\n          残り {len(todo)} / {len(cases)} 症例 -> {out_dir}")
    if not todo:
        return

    pred = nnUNetPredictor(device=torch.device(device), verbose=False, allow_tqdm=True)
    # use_mirroring defaults to True and picks up the checkpoint's own
    # inference_allowed_mirroring_axes, which is (0,1,2) for the official model.
    # perform_actual_validation used the same axes, so our stored predictions match.
    pred.initialize_from_trained_model_folder(model_dir, use_folds=(fold,),
                                              checkpoint_name="checkpoint_final.pth")
    pred.predict_from_files(
        [channel_files(dataset, c) for c in todo],
        [join(out_dir, c) for c in todo],
        save_probabilities=False, overwrite=False,
        num_processes_preprocessing=2, num_processes_segmentation_export=2,
    )


def evaluate(folder_pred, dataset, official_model, out_json):
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2
    compute_metrics_on_folder2(
        join(dataset, "labelsTr"), folder_pred,
        join(official_model, "dataset.json"), join(official_model, "plans.json"),
        output_file=out_json, chill=True,
    )
    return out_json


def summarise(name, path):
    d = json.load(open(path))
    per = d["metric_per_case"]
    lab = list(per[0]["metrics"].keys())[0]

    def get(x, m):
        return x["metrics"][lab].get(m)

    def isnan(v):
        return v is None or (isinstance(v, float) and math.isnan(v))

    tot = len(per)
    dropped = sum(1 for x in per if isnan(get(x, "Dice")))
    free = [x for x in per if get(x, "n_ref") == 0]
    free_zero = sum(1 for x in free if not isnan(get(x, "Dice")) and get(x, "Dice") == 0)
    mean = d.get("mean", {}).get(lab, {}).get("Dice")
    pos = [get(x, "Dice") for x in per if get(x, "n_ref") != 0 and not isnan(get(x, "Dice"))]
    pos_mean = sum(pos) / len(pos) if pos else float("nan")
    print(f"\n--- {name} ---")
    print(f"  症例 {tot} / 平均に入った {tot - dropped} / 除外 {dropped}")
    print(f"  病変なし {len(free)}  うち除外 {len(free) - free_zero}  うち Dice=0 計上 {free_zero}")
    print(f"  reported mean         {mean}")
    print(f"  病変ありのみの平均     {pos_mean:.4f}  (n={len(pos)})")
    return dict(name=name, total=tot, entered=tot - dropped, free=len(free),
                free_zero=free_zero, mean=mean, pos_mean=pos_mean, n_pos=len(pos))


def main():
    args = parse_args()
    cases = val_cases(args.preprocessed, args.fold)
    if not check(args, cases):
        sys.exit(1)
    if args.dry_run:
        print("\n--dry-run なのでここで終了。問題なければ --dry-run を外して再実行してください。")
        return

    os.makedirs(args.out, exist_ok=True)
    off_pred = join(args.out, "official_pred")
    predict(args.official, args.fold, cases, args.dataset, off_pred, args.device)
    rows = [summarise("公式baseline", evaluate(off_pred, args.dataset, args.official,
                                               join(args.out, "summary_official.json")))]

    ours_val = join(args.results, args.ours_dirname, f"fold_{args.fold}", "validation")
    if args.also_ours:
        ours_val = join(args.out, "ours_pred")
        predict(join(args.results, args.ours_dirname), args.fold, cases, args.dataset,
                ours_val, args.device)
    rows.append(summarise("自前 click修正版", evaluate(ours_val, args.dataset, args.official,
                                                   join(args.out, "summary_ours.json"))))

    print("\n=== 同一評価コード・同一323例での比較 ===")
    print(f"{'model':<18}{'entered':>9}{'mean':>10}{'病変ありのみ':>14}")
    for r in rows:
        print(f"{r['name']:<18}{r['entered']:>9}{r['mean']:>10.4f}{r['pos_mean']:>14.4f}")
    print(f"\n出力: {args.out}")


if __name__ == "__main__":
    main()
