import os
import json
import shutil
import argparse
import numpy as np
import SimpleITK as sitk
import subprocess
import traceback

from simulate_scribbles import simulate_scribble_from_label

sitk.ProcessObject_SetGlobalWarningDisplay(False)


def dice_score(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    intersection = np.sum(pred * gt)
    denom = np.sum(pred) + np.sum(gt)

    if denom == 0:
        return 1.0

    return 2.0 * intersection / denom

def detection_matching_metric(pred, gt):
    # TODO - add the implementation here
    return 0

def log_error(case, iteration, message):
    with open(error_log_file, "a") as f:
        f.write(f"CASE: {case} | ITER: {iteration} | ERROR: {message}\n")


parser = argparse.ArgumentParser()
parser.add_argument("--input_cases", type=str)
parser.add_argument("--result_dir", type=str)
parser.add_argument("--input_interface", type=str)
parser.add_argument("--strategy", required=True,
                    choices=["centerline", "random", "boundary"])
args = parser.parse_args()

input_cases = args.input_cases
input_interface = args.input_interface
result_dir = args.result_dir
strategy = args.strategy

# We assume the CT images are stored with a suffix _0000 and PET (SUV) with _0001 in the nnUNet format
cts = sorted([os.path.join(input_cases, 'images', el)
              for el in os.listdir(os.path.join(input_cases, 'images')) if '_0000' in el])
pets = sorted([os.path.join(input_cases, 'images', el)
               for el in os.listdir(os.path.join(input_cases, 'images')) if '_0001' in el])
labels = sorted([os.path.join(input_cases, 'labels', el)
                 for el in os.listdir(os.path.join(input_cases, 'labels'))])

os.makedirs(result_dir, exist_ok=True)
error_log_file = os.path.join(result_dir, "error_log.txt")
with open(error_log_file, "w") as f:
    f.write("=== ERROR LOG ===\n")

case_dict = {}
output_dice_file = os.path.join(result_dir, "dice_scores.json")

max_iters = 2

# =========================
# Case loop
# =========================
for it_case, (ct, pet, label) in enumerate(zip(cts, pets, labels)):

    tag = os.path.basename(ct).split('.nii.gz')[0]
    print(f"\n================ CASE {tag} ================\n")

    case_dict[tag] = []

    # ===========================================================================
    # Clean input interface - make sure there is only one case per inference step
    # ===========================================================================
    ct_dir = os.path.join(input_interface, 'input', 'images', 'ct')
    pet_dir = os.path.join(input_interface, 'input', 'images', 'pet')
    seg_dir = os.path.join(input_interface, 'output', 'images', 'tumor-lesion-segmentation')

    for d in [ct_dir, pet_dir, seg_dir]:
        if os.path.exists(d):
            for f in os.listdir(d):
                try:
                    os.remove(os.path.join(d, f))
                except Exception:
                    pass 

    try:
        # Load images
        ct_img = sitk.ReadImage(ct)
        pet_img = sitk.ReadImage(pet)
        label_img = sitk.ReadImage(label)

        # Write inputs
        ct_out = os.path.join(input_interface, 'input', 'images', 'ct', f"case_{tag}.mha")
        pet_out = os.path.join(input_interface, 'input', 'images', 'pet', f"case_{tag}.mha")

        sitk.WriteImage(ct_img, ct_out)
        sitk.WriteImage(pet_img, pet_out)

        output_json = os.path.join(input_interface, 'input', 'lesion-clicks.json')
        output_seg = os.path.join(input_interface, 'output', 'images', 'tumor-lesion-segmentation')
        prev_dice = None  # store last valid dice
        empty_gt = False
        # =========================
        # Interaction loop
        # =========================
        for it in range(max_iters):
            print(f'Interactive iteration {it} starting...\n')

            try:
                if it == 0:
                    data = {
                        "tumor": [],
                        "background": []
                    }

                else:
                    gt = sitk.GetArrayFromImage(label_img)
                    empty_gt = (np.sum(gt) == 0)
                    if empty_gt:
                        print("Empty GT detected → skipping inference, reusing Dice from iteration 0")
                        dice = prev_dice if prev_dice is not None else 1.0
                    else:
                        seg_path = os.path.join(output_seg, f"case_{tag}.mha")

                        if not os.path.exists(seg_path):
                            raise FileNotFoundError(f"Missing segmentation: {seg_path}")

                        seg_img = sitk.ReadImage(seg_path)

                        with open(output_json, 'r') as f:
                            data = json.load(f)

                        pred = sitk.GetArrayFromImage(seg_img)


                        if pred.shape != gt.shape:
                            raise ValueError("Shape mismatch between prediction and GT")

                        error_map = np.abs(pred - gt).astype(np.uint8)

                        scribbles, label_cls = simulate_scribble_from_label(error_map, strategy)

                        if label_cls:
                            data['tumor'] += scribbles
                        else:
                            data['background'] += scribbles

                # Update input scribbles
                with open(output_json, "w") as f:
                    json.dump(data, f)

                if not (it > 0 and empty_gt):
                    # Run inference
                    subprocess.run(
                        "bash nnunet-baseline/test.sh",
                        shell=True,
                        timeout=600,
                        check=True
                    )

                    # Load prediction
                    seg_path = os.path.join(output_seg, f"case_{tag}.mha")

                    if not os.path.exists(seg_path):
                        raise FileNotFoundError("Prediction file not created")

                    seg_img = sitk.ReadImage(seg_path)

                    pred = sitk.GetArrayFromImage(seg_img)
                    gt = sitk.GetArrayFromImage(label_img)

                    if pred.shape != gt.shape:
                        raise ValueError("Shape mismatch after inference")

                    dice = dice_score(pred, gt)
                    dmm = detection_matching_metric(pred, gt)

            except Exception as e:
                print(f"[WARNING] Iteration {it} failed for case {tag}: {e}")
                print(traceback.format_exc())

                log_error(tag, it, str(e))  

                dice = 0.0
                dmm = 0.0

            prev_dice = float(dice)
            # Always record Dice
            case_dict[tag].append({
                "iteration": it,
                "dice": float(dice),
                "dmm": float(dmm)
            })

            print(f'\nDice@{it} = {dice}\n')
            print(f'\DMM@{it} = {dmm}\n')

            # Safe copy of result
            try:
                shutil.copy(
                    os.path.join(output_seg, f"case_{tag}.mha"),
                    os.path.join(result_dir, f"case_{tag}_{it}.mha")
                )
            except Exception:
                pass

        # =========================
        # Cleanup 
        # =========================
        for path in [
            ct_out,
            pet_out,
            os.path.join(output_seg, f"case_{tag}.mha")
        ]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    except Exception as e:
        print(f"[ERROR] Case {tag} completely failed: {e}")
        print(traceback.format_exc())

        # Fill all iterations with 0
        case_dict[tag] = [{"iteration": i, "dice": 0.0} for i in range(6)]
        case_dict[tag] = [{"iteration": i, "dmm": 0.0} for i in range(6)]

    # =========================
    # Save progress after each case
    # =========================
    with open(output_dice_file, "w") as f:
        json.dump(case_dict, f, indent=4)


print("\n✅ All cases processed. Dice scores saved.\n")

# -------------------------
print("Computing AUC metrics.")
with open(output_dice_file, "r") as f:
    data = json.load(f)

auc_results = {}

# -------------------------
# Compute AUC per case
# -------------------------
for case_id, records in data.items():

    # sort by iteration (important!)
    records = sorted(records, key=lambda x: x["iteration"])

    iterations = np.array([r["iteration"] for r in records], dtype=float)
    dice = np.array([r["dice"] for r in records], dtype=float)
    # TODO add dmm
    # dmm = np.array([r["dmm"] for r in records], dtype=float)


    auc = np.trapz(dice, iterations)

    auc_results[case_id] = {
        "auc": float(auc),  
        # "dmm": float(dmm)
    }

# -------------------------
# Save output JSON
# -------------------------
auc_output_file = output_dice_file.replace('.json', '_AUC.json')
with open(auc_output_file, "w") as f:
    json.dump(auc_results, f, indent=4)

print("Saved AUC results to:", auc_output_file)