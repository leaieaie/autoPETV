import json
import os
import shutil
import subprocess
from pathlib import Path
import SimpleITK
import torch

from utils import save_click_heatmaps

class Autopet_baseline:

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        # according to the specified grand-challenge interfaces
        self.input_path = "/input/"
        # according to the specified grand-challenge interfaces
        self.output_path = "/output/images/tumor-lesion-segmentation/"
        self.nii_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs"
        )
        # Second input tree, identical CT/PET but point-marker click channels.
        # The two ensemble members were trained with different click encodings, so
        # they cannot share one input folder (see load_inputs).
        self.nii_path_point = self.nii_path + "_point"
        self.lesion_click_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/clicksTs"
        )
        self.result_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result"
        )
        self.nii_seg_file = "TCIA_001.nii.gz"
        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def gc_to_swfastedit_format(self, gc_json_path, swfast_json_path):
        with open(gc_json_path, 'r') as f:
            gc_dict = json.load(f)
        swfast_dict = {
            "tumor": [],
            "background": []
        }

        for point in gc_dict.get("points", []):
            if point["name"] == "tumor":
                swfast_dict["tumor"].append(point["point"])
            elif point["name"] == "background":
                swfast_dict["background"].append(point["point"])
        with open(swfast_json_path, 'w') as f:
            json.dump(swfast_dict, f)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print("Available: " + str(is_available))
        print(f"Device count: {torch.cuda.device_count()}")
        if is_available:
            print(f"Current device: {torch.cuda.current_device()}")
            print("Device name: " + torch.cuda.get_device_name(0))
            print(
                "Device memory: "
                + str(torch.cuda.get_device_properties(0).total_memory)
            )

    def check_models(self):
        """Fail loudly if a model folder is missing or holds a placeholder.

        The container's model folders have been overwritten by hand more than once,
        and a checkpoint of the wrong model is indistinguishable from the right one
        by path or by size. Printing what is actually there makes a mis-built image
        visible in the run log instead of silently producing a worse submission.
        """
        base = "/opt/algorithm/nnUNet_results/Dataset998_AutoPETV"
        ok = True
        for plans in ("nnUNetPlans", "nnUNetResEncUNetMPlans"):
            folder = os.path.join(base, f"nnUNetTrainer__{plans}__3d_fullres")
            ckpt = os.path.join(folder, "fold_0", "checkpoint_final.pth")
            if not os.path.isfile(ckpt):
                print(f"[models] MISSING {ckpt}")
                ok = False
                continue
            size_mb = os.path.getsize(ckpt) / 1e6
            norm = "?"
            try:
                with open(os.path.join(folder, "plans.json")) as f:
                    cfg = json.load(f)["configurations"]["3d_fullres"]
                norm = ",".join(s.replace("Normalization", "") for s in cfg["normalization_schemes"])
            except Exception as e:
                print(f"[models] plans.json を読めない: {e}")
                ok = False
            print(f"[models] {plans}: checkpoint {size_mb:.0f} MB / norm {norm}")
            if size_mb < 100:
                print(f"[models] checkpoint が小さすぎる ({size_mb:.0f} MB) = LFS ポインタの可能性")
                ok = False
        if not ok:
            raise RuntimeError("model folders are not set up correctly; see [models] lines above")

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/ct/", ct_mha),
            os.path.join(self.nii_path, "TCIA_001_0000.nii.gz"),
        )
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/pet/", pet_mha),
            os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"),
        )

        json_file = os.path.join(self.input_path, "lesion-clicks.json")
        print(f"json_file: {json_file}")
        self.gc_to_swfastedit_format(json_file, os.path.join(self.lesion_click_path, "TCIA_001_clicks.json"))

        click_file = os.listdir(self.lesion_click_path)[0]
        if click_file:
            with open(os.path.join(self.lesion_click_path, click_file), 'r') as f:
                clicks = json.load(f)
            pet_ref = os.path.join(self.nii_path, "TCIA_001_0001.nii.gz")

            # The two ensemble members disagree on how a click looks:
            #   official baseline : single-voxel markers, ZScore on ch2/ch3
            #   ResEncM OTF       : exp(-d/tau) EDT field, NoNormalization on ch2/ch3
            # Feeding either model the other's encoding puts it far outside its
            # training distribution, so each gets its own input tree. CT and PET are
            # identical between the two and are hard-linked rather than re-converted.
            os.makedirs(self.nii_path_point, exist_ok=True)
            for ch in ("0000", "0001"):
                src = os.path.join(self.nii_path, f"TCIA_001_{ch}.nii.gz")
                dst = os.path.join(self.nii_path_point, f"TCIA_001_{ch}.nii.gz")
                if os.path.exists(dst):
                    os.remove(dst)
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copyfile(src, dst)

            os.environ["CLICK_ENCODING"] = "edt"
            save_click_heatmaps(clicks, self.nii_path, pet_ref)
            os.environ["CLICK_ENCODING"] = "point"
            save_click_heatmaps(clicks, self.nii_path_point, pet_ref)

        print("edt  :", sorted(os.listdir(self.nii_path)))
        print("point:", sorted(os.listdir(self.nii_path_point)))

        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(
            os.path.join(self.result_path, self.nii_seg_file),
            os.path.join(self.output_path, uuid + ".mha"),
        )
        print("Output written to: " + os.path.join(self.output_path, uuid + ".mha"))

    def predict(self):
        """
        Ensemble of the official baseline and our ResEncM on-the-fly-EDT model.

        Measured on the same fold-0 validation split with one evaluator, the official
        baseline scores 0.8140 lesion-positive Dice against 0.7733 for our best
        retrained model, and it reproduces 0.7800 / 0.7554 on the preliminary
        leaderboard from this same container, so it is the stronger member. The
        ResEncM OTF model is our best on lesion detection (LB F1 0.7059 against the
        baseline's 0.7554 Dice-side strength); the ranking averages the Dice and F1
        positions, so a member that helps detection can move the total even when it
        is weaker overall. Ensembling is also the only change that has ever improved
        our combined score.
        """
        print("nnUNet ensemble segmentation starting!")
        d1 = self.result_path + "_m1"   # official baseline  (point markers + ZScore)
        d2 = self.result_path + "_m2"   # ResEncM OTF        (EDT + NoNormalization)
        os.makedirs(d1, exist_ok=True)
        os.makedirs(d2, exist_ok=True)

        subprocess.run(
            f"nnUNetv2_predict -i {self.nii_path_point} -o {d1} -d 998 -c 3d_fullres -f 0 "
            f"-p nnUNetPlans --save_probabilities --disable_tta",
            shell=True, check=True,
        )
        subprocess.run(
            f"nnUNetv2_predict -i {self.nii_path} -o {d2} -d 998 -c 3d_fullres -f 0 "
            f"-p nnUNetResEncUNetMPlans --save_probabilities --disable_tta",
            shell=True, check=True,
        )
        # average the two softmax volumes and write the final segmentation
        subprocess.run(
            f"nnUNetv2_ensemble -i {d1} {d2} -o {self.result_path}",
            shell=True, check=True,
        )
        print("Ensemble prediction finished")

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        self.check_models()
        print("Start processing")
        uuid = self.load_inputs()
        print("Start prediction")
        self.predict()
        print("Start output writing")
        self.write_outputs(uuid)


if __name__ == "__main__":
    print("START")
    Autopet_baseline().process()
