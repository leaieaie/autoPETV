"""
Diagnostic: native 6-step interactive loop for the OTF-click model.

Purpose: understand WHY the on-the-fly EDT-click submission regressed on the
hidden test set (AUC-Dice 0.7525 / AUC-DMM 0.6622 vs baseline 0.7800 / 0.7554),
especially the large F1/DMM drop. We replay the official 6-step corrective loop
on a labeled case and print, per step, Dice + DMM + the foreground voxel count,
so we can see:
  * does Dice climb as foreground clicks accumulate (is the model responsive)?
  * when background clicks appear (later steps), does the foreground collapse
    (over-suppression -> the suspected cause of the F1/DMM drop)?

Differences vs interactive/interactive_loop.py and the baseline native script:
  * click channels (ch2/ch3) are encoded with the SAME EDT field used in training
    (click_encoding.encode_clicks_edt), not the old single-voxel point marker.
  * points nnUNet at the OTF model folder (checkpoint with trainer_name patched to
    'nnUNetTrainer', plans.json with ch2/ch3 = NoNormalization).

Reuses the official modules: interactive/simulate_scribbles.py and metrics.py.

Defaults target PC-C (okudalab8). Run:
  <autopetv-python> scripts/diag_interactive_otf.py
"""
import os
import sys
import json
import shutil
import argparse
import subprocess

import numpy as np
import nibabel as nib

# PC-C defaults
FORK_DEFAULT = r"C:\Users\OkudaLab08\autoPETV"
RESULTS_DEFAULT = r"C:\Users\OkudaLab08\autoPETV\nnunet-baseline\nnUNet_results"
PREDICT_EXE_DEFAULT = r"C:\miniconda\envs\autopetv\Scripts\nnUNetv2_predict.exe"


def safe_simulate(region, strategy, simulate_fn):
    if region.sum() == 0:
        return [], 0
    res = simulate_fn(region.astype(np.uint8), strategy)
    if isinstance(res, tuple) and len(res) == 3:
        return res[0], int(res[2])
    return [], 0


def build_inputs(case, ct_path, pet_path, clicks, in_dir, encode_fn):
    os.makedirs(in_dir, exist_ok=True)
    for f in os.listdir(in_dir):
        os.remove(os.path.join(in_dir, f))
    shutil.copy(ct_path, os.path.join(in_dir, f"{case}_0000.nii.gz"))
    shutil.copy(pet_path, os.path.join(in_dir, f"{case}_0001.nii.gz"))
    pet = nib.load(pet_path)
    shape, aff, hdr = pet.shape, pet.affine, pet.header
    fg = encode_fn(clicks["tumor"], shape)
    bg = encode_fn(clicks["background"], shape)
    nib.save(nib.Nifti1Image(fg.astype(np.float32), aff, hdr), os.path.join(in_dir, f"{case}_0002.nii.gz"))
    nib.save(nib.Nifti1Image(bg.astype(np.float32), aff, hdr), os.path.join(in_dir, f"{case}_0003.nii.gz"))


def run_predict(predict_exe, in_dir, out_dir, results_dir):
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env["nnUNet_results"] = results_dir
    env.setdefault("nnUNet_raw", os.path.join(os.path.dirname(results_dir), "_raw"))
    env.setdefault("nnUNet_preprocessed", os.path.join(os.path.dirname(results_dir), "_pre"))
    subprocess.run(
        [predict_exe, "-i", in_dir, "-o", out_dir,
         "-d", "998", "-c", "3d_fullres", "-f", "0", "--disable_tta"],
        check=True, env=env,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", default=FORK_DEFAULT)
    ap.add_argument("--results", default=RESULTS_DEFAULT)
    ap.add_argument("--predict_exe", default=PREDICT_EXE_DEFAULT)
    ap.add_argument("--case", default="psma_ffcaa75377465b37_2018-03-04")
    ap.add_argument("--strategy", default="random", choices=["centerline", "random", "boundary"])
    ap.add_argument("--max_iters", type=int, default=6)
    ap.add_argument("--encoding", default="edt", choices=["edt", "point"])
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.fork)                                 # metrics.py
    sys.path.insert(0, os.path.join(args.fork, "interactive"))    # simulate_scribbles.py
    sys.path.insert(0, os.path.join(args.fork, "nnunet-baseline"))  # click_encoding.py
    from simulate_scribbles import simulate_scribble_from_label, generate_gaussian_heatmap
    from metrics import MetricEvaluator
    from click_encoding import encode_clicks_edt

    if args.encoding == "edt":
        encode_fn = lambda coords, shape: encode_clicks_edt(coords, shape)
    else:
        encode_fn = lambda coords, shape: generate_gaussian_heatmap(coords, shape, sigma=0)

    case = args.case
    ct_path = os.path.join(args.fork, "test", "images", f"{case}_0000.nii.gz")
    pet_path = os.path.join(args.fork, "test", "images", f"{case}_0001.nii.gz")
    gt_path = os.path.join(args.fork, "test", "labels", f"{case}.nii.gz")
    workdir = args.workdir or os.path.join(args.fork, "test", "diag_otf")
    os.makedirs(workdir, exist_ok=True)

    gt_img = nib.load(gt_path)
    gt = (gt_img.get_fdata() > 0).astype(np.uint8)
    spacing = [float(z) for z in gt_img.header.get_zooms()]
    gt_vox = int(gt.sum())
    evaluator = MetricEvaluator()

    clicks = {"tumor": [], "background": []}
    records = []
    last_pred = None

    print(f"=== OTF diagnostic | case={case} strategy={args.strategy} encoding={args.encoding} ===")
    print(f"GT foreground voxels = {gt_vox}")

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

        in_dir = os.path.join(workdir, f"iter_{it}", "in")
        out_dir = os.path.join(workdir, f"iter_{it}", "out")
        build_inputs(case, ct_path, pet_path, clicks, in_dir, encode_fn)
        run_predict(args.predict_exe, in_dir, out_dir, args.results)

        pred = (nib.load(os.path.join(out_dir, f"{case}.nii.gz")).get_fdata() > 0).astype(np.uint8)
        pred_vox = int(pred.sum())
        m = evaluator(pred, gt, f"{case}_{it}", spacing=spacing)
        dice = float(m["dsc"]) if not np.isnan(m["dsc"]) else 0.0
        dmm = float(m["f1"]) if not np.isnan(m["f1"]) else 0.0
        records.append({
            "iteration": it, "dice": dice, "dmm": dmm, "pred_vox": pred_vox,
            "n_fg_clicks": len(clicks["tumor"]), "n_bg_clicks": len(clicks["background"]),
        })
        print(f"[iter {it}] Dice={dice:.4f}  DMM={dmm:.4f}  pred_vox={pred_vox:>7d}  "
              f"fg_clicks={len(clicks['tumor'])} bg_clicks={len(clicks['background'])}", flush=True)
        last_pred = pred

    iters = np.array([r["iteration"] for r in records], dtype=float)
    auc_dice = float(np.trapz([r["dice"] for r in records], iters))
    auc_dmm = float(np.trapz([r["dmm"] for r in records], iters))

    summary = {"case": case, "strategy": args.strategy, "encoding": args.encoding,
               "gt_vox": gt_vox, "records": records,
               "auc_dice": auc_dice, "auc_dmm": auc_dmm}
    out_json = os.path.join(workdir, f"diag_otf_{args.encoding}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==== SUMMARY (OTF, encoding=%s) ====" % args.encoding)
    print(f"GT_vox={gt_vox}")
    for r in records:
        print(f"  iter {r['iteration']}: Dice={r['dice']:.4f}  DMM={r['dmm']:.4f}  "
              f"pred_vox={r['pred_vox']:>7d}  (fg={r['n_fg_clicks']}, bg={r['n_bg_clicks']})")
    print(f"AUC-Dice={auc_dice:.4f}  AUC-DMM={auc_dmm:.4f}")
    print("Baseline reference (point enc): iter0 Dice0.870 .. iter5 Dice0.937, DMM 0.778 flat, "
          "AUC-Dice 4.56 / AUC-DMM 3.89")
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
