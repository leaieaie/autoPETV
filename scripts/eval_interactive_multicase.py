"""
Aggregated interactive 6-step evaluation over many validation cases.

Everything so far was judged on a single PSMA case, where DMM is coarsely
quantized and cannot stand in for the leaderboard's aggregate F1. Now that all
1611 cases are present in original geometry under D:\autopet\dataset, we can
replay the official 6-step corrective loop over a sample of the fold-0
validation split and average per-step Dice + DMM -- a local estimate that
tracks the hidden-set metric, so models can be compared without spending a
leaderboard submission.

The model is loaded ONCE via nnUNetPredictor and reused across every case and
step (the CLI reloaded it on each of ~60+ calls). Click simulation, encoding and
metrics are identical to scripts/diag_interactive_otf.py.

Only cases with a non-empty label are sampled, so Dice/DMM are defined; lesion-
free cases (where the interesting question is click-free false positives) are a
separate check.

Run on PC-C, e.g.:
  <autopetv-python> scripts/eval_interactive_multicase.py \
      --trainer nnUNetTrainer_ClickDropout --n 15
"""
import os
import sys
import json
import shutil
import argparse
import tempfile

import numpy as np
import nibabel as nib
import torch

FORK_DEFAULT = r"C:\Users\OkudaLab08\autoPETV"
RESULTS_DEFAULT = r"D:\autopet\nnUNet\results"
IMAGES_DEFAULT = r"D:\autopet\dataset\Dataset998_AutoPETV\imagesTr"
LABELS_DEFAULT = r"D:\autopet\dataset\Dataset998_AutoPETV\labelsTr"
SPLITS_DEFAULT = r"D:\autopet\nnUNet\preprocessed\Dataset998_AutoPETV\splits_final.json"


def safe_simulate(region, strategy, simulate_fn):
    if region.sum() == 0:
        return [], 0
    res = simulate_fn(region.astype(np.uint8), strategy)
    if isinstance(res, tuple) and len(res) == 3:
        return res[0], int(res[2])
    return [], 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", default=FORK_DEFAULT)
    ap.add_argument("--results", default=RESULTS_DEFAULT)
    ap.add_argument("--trainer", default="nnUNetTrainer")
    ap.add_argument("--plans", default="nnUNetPlans")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--model_dirname", default=None,
                    help="exact results subfolder name; overrides trainer__plans__config")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--images", default=IMAGES_DEFAULT)
    ap.add_argument("--labels", default=LABELS_DEFAULT)
    ap.add_argument("--splits", default=SPLITS_DEFAULT)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=15, help="validation cases to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_gt_vox", type=int, default=50)
    ap.add_argument("--strategy", default="random", choices=["centerline", "random", "boundary"])
    ap.add_argument("--max_iters", type=int, default=6)
    ap.add_argument("--encoding", default="point", choices=["edt", "point"])
    args = ap.parse_args()

    sys.path.insert(0, args.fork)
    sys.path.insert(0, os.path.join(args.fork, "interactive"))
    sys.path.insert(0, os.path.join(args.fork, "nnunet-baseline"))
    from simulate_scribbles import simulate_scribble_from_label, generate_gaussian_heatmap
    from metrics import MetricEvaluator
    from click_encoding import encode_clicks_edt

    if args.encoding == "edt":
        encode_fn = lambda coords, shape: encode_clicks_edt(coords, shape)
    else:
        encode_fn = lambda coords, shape: generate_gaussian_heatmap(coords, shape, sigma=0)

    # ---- pick validation cases with a non-empty label ----
    val = json.load(open(args.splits))[args.fold]["val"]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(val)
    cases = []
    for c in val:
        lab_path = os.path.join(args.labels, f"{c}.nii.gz")
        if not os.path.exists(lab_path):
            continue
        vox = int((np.asanyarray(nib.load(lab_path).dataobj) > 0).sum())
        if vox >= args.min_gt_vox:
            cases.append((c, vox))
        if len(cases) >= args.n:
            break
    print(f"fold {args.fold} val: {len(val)} cases; using {len(cases)} with GT>= {args.min_gt_vox} vox")
    print(f"model: {args.trainer}__{args.plans}__{args.config}  ckpt={args.checkpoint}  encoding={args.encoding}\n")

    # ---- load the model ONCE ----
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        device=torch.device("cuda", 0), verbose=False,
        verbose_preprocessing=False, allow_tqdm=False,
    )
    dirname = args.model_dirname or f"{args.trainer}__{args.plans}__{args.config}"
    model_folder = os.path.join(args.results, "Dataset998_AutoPETV", dirname)
    predictor.initialize_from_trained_model_folder(
        model_folder, use_folds=(args.fold,), checkpoint_name=args.checkpoint)

    evaluator = MetricEvaluator()
    tmp = tempfile.mkdtemp(prefix="mc_eval_")
    in_dir, out_dir = os.path.join(tmp, "in"), os.path.join(tmp, "out")

    all_dice = np.zeros((len(cases), args.max_iters))
    all_dmm = np.zeros((len(cases), args.max_iters))

    for ci, (case, gtvox) in enumerate(cases):
        ct = os.path.join(args.images, f"{case}_0000.nii.gz")
        pet = os.path.join(args.images, f"{case}_0001.nii.gz")
        gt_img = nib.load(os.path.join(args.labels, f"{case}.nii.gz"))
        gt = (gt_img.get_fdata() > 0).astype(np.uint8)
        spacing = [float(z) for z in gt_img.header.get_zooms()]
        pet_img = nib.load(pet)
        shape, aff, hdr = pet_img.shape, pet_img.affine, pet_img.header

        clicks = {"tumor": [], "background": []}
        last_pred = None
        line = [f"[{ci+1}/{len(cases)}] {case[:34]:<34} gt={gtvox:>6d}"]

        for it in range(args.max_iters):
            if it > 0 and last_pred is not None:
                overseg = ((last_pred > 0) & (gt == 0)).astype(np.uint8)
                underseg = ((last_pred == 0) & (gt > 0)).astype(np.uint8)
                scr_bg, fp = safe_simulate(overseg, args.strategy, simulate_scribble_from_label)
                scr_fg, fn = safe_simulate(underseg, args.strategy, simulate_scribble_from_label)
                if fp <= fn:
                    clicks["tumor"] += scr_fg
                else:
                    clicks["background"] += scr_bg

            if os.path.isdir(in_dir):
                shutil.rmtree(in_dir)
            os.makedirs(in_dir)
            shutil.copy(ct, os.path.join(in_dir, f"{case}_0000.nii.gz"))
            shutil.copy(pet, os.path.join(in_dir, f"{case}_0001.nii.gz"))
            fg = encode_fn(clicks["tumor"], shape).astype(np.float32)
            bg = encode_fn(clicks["background"], shape).astype(np.float32)
            nib.save(nib.Nifti1Image(fg, aff, hdr), os.path.join(in_dir, f"{case}_0002.nii.gz"))
            nib.save(nib.Nifti1Image(bg, aff, hdr), os.path.join(in_dir, f"{case}_0003.nii.gz"))

            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir)
            predictor.predict_from_files(
                in_dir, out_dir, save_probabilities=False, overwrite=True,
                num_processes_preprocessing=1, num_processes_segmentation_export=1,
            )
            pred = (nib.load(os.path.join(out_dir, f"{case}.nii.gz")).get_fdata() > 0).astype(np.uint8)
            m = evaluator(pred, gt, f"{case}_{it}", spacing=spacing)
            d = float(m["dsc"]) if not np.isnan(m["dsc"]) else 0.0
            f = float(m["f1"]) if not np.isnan(m["f1"]) else 0.0
            all_dice[ci, it] = d
            all_dmm[ci, it] = f
            line.append(f"{d:.3f}/{f:.3f}")
            last_pred = pred
        print("  ".join(line), flush=True)

    shutil.rmtree(tmp, ignore_errors=True)

    steps = np.arange(args.max_iters, dtype=float)
    mean_dice = all_dice.mean(0)
    mean_dmm = all_dmm.mean(0)
    auc_dice = float(np.trapz(mean_dice, steps))
    auc_dmm = float(np.trapz(mean_dmm, steps))

    print("\n==== AGGREGATE (%d cases, %s, encoding=%s) ====" % (len(cases), dirname, args.encoding))
    print("step :  " + "  ".join(f"{i}" for i in range(args.max_iters)))
    print("Dice :  " + "  ".join(f"{v:.3f}" for v in mean_dice))
    print("DMM  :  " + "  ".join(f"{v:.3f}" for v in mean_dmm))
    print(f"\nAUC-Dice = {auc_dice:.4f}   AUC-DMM = {auc_dmm:.4f}")
    print("Leaderboard baseline (official weights, HIDDEN set): AUC-Dice 0.7800 / AUC-F1 0.7554")
    print("  (note: local trapz AUC is area over 6 steps, ~5x the leaderboard's mean scale;")
    print("   compare models to each other here, use the leaderboard for the absolute bar.)")

    out = {"trainer": dirname, "encoding": args.encoding, "n_cases": len(cases),
           "cases": [c for c, _ in cases], "mean_dice": mean_dice.tolist(),
           "mean_dmm": mean_dmm.tolist(), "auc_dice": auc_dice, "auc_dmm": auc_dmm}
    dst = os.path.join(args.fork, "test", f"mc_eval_{dirname}.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(f"saved: {dst}")


if __name__ == "__main__":
    main()
