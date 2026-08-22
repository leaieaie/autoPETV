import json
import os
import subprocess
import SimpleITK
import torch

from utils import save_click_heatmaps


def _env(name, default):
    return os.environ.get(name, default)


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
        self.lesion_click_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/clicksTs"
        )
        self.result_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result"
        )
        self.nii_seg_file = "TCIA_001.nii.gz"

        # Inference knobs, set from the Dockerfile so one image per variant can be
        # built without touching code. Defaults reproduce the official baseline
        # exactly: checkpoint_final, step 0.5, mirroring off, point-marker clicks.
        self.plans = _env("NNUNET_PLANS", "nnUNetPlans")
        self.checkpoint = _env("NNUNET_CHK", "checkpoint_final.pth")
        self.step_size = _env("NNUNET_STEP", "0.5")
        self.tta = _env("NNUNET_TTA", "off").lower() in ("1", "on", "true", "yes")
        os.environ.setdefault("CLICK_ENCODING", "point")

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

    def check_model(self):
        """Say in the log which weights and settings this image actually runs.

        The model folder in this repository has been overwritten by hand more than
        once, and a checkpoint of the wrong model is indistinguishable from the
        right one by path or by size, so the run log is the only place a mis-built
        image becomes visible. mean_fg_dice is the training history stored inside
        the checkpoint; for the official baseline it is 1000 epochs ending 0.7043
        with a best of 0.8349.
        """
        folder = f"/opt/algorithm/nnUNet_results/Dataset998_AutoPETV/nnUNetTrainer__{self.plans}__3d_fullres"
        ckpt = os.path.join(folder, "fold_0", self.checkpoint)
        if not os.path.isfile(ckpt):
            raise RuntimeError(f"checkpoint がない: {ckpt}")
        try:
            with open(os.path.join(folder, "plans.json")) as f:
                cfg = json.load(f)["configurations"]["3d_fullres"]
            norm = ",".join(s.replace("Normalization", "") for s in cfg["normalization_schemes"])
        except Exception as e:
            raise RuntimeError(f"plans.json を読めない: {e}")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        hist = (ck.get("logging") or {}).get("mean_fg_dice") or []
        tail = f" best={max(hist):.4f} last={hist[-1]:.4f}" if hist else ""
        print(f"[model] {self.plans} / {self.checkpoint} / {os.path.getsize(ckpt)/1e6:.0f} MB")
        print(f"[model] norm={norm} trainer={ck.get('trainer_name')} epochs={len(hist)}{tail}")
        print(f"[model] step_size={self.step_size} tta={'on' if self.tta else 'off'} "
              f"clicks={os.environ.get('CLICK_ENCODING')}")

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
            save_click_heatmaps(clicks, self.nii_path,
                                os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"),
                                )
        print(os.listdir(self.nii_path))

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
        """Official baseline weights, with the inference settings under our control.

        Everything we retrained lost to these weights: 0.8140 against 0.7733
        lesion-positive Dice on the same fold-0 validation split with one
        evaluator, and 0.7800/0.7554 on the leaderboard from this container.
        Ensembling them with a weaker model lost on both metrics, so what is left
        to move is how these weights are run. The three knobs below were never
        varied on this model:

          -chk        the container has always used checkpoint_final, but the
                      official run ends at mean_fg_dice 0.7043 with a best of
                      0.8349, so checkpoint_best is a different model in practice
          -step_size  0.5 is the nnU-Net default; denser sliding windows usually
                      buy a little Dice, and one case predicted in 13 s locally,
                      well inside the 600 s per-iteration budget
          TTA         these weights carry inference_allowed_mirroring_axes=(0,1,2)
                      but the container disables it. The earlier TTA experiment
                      was run on our own model, not on these weights.
        """
        cmd = (
            f"nnUNetv2_predict -i {self.nii_path} -o {self.result_path} -d 998 -c 3d_fullres -f 0 "
            f"-p {self.plans} -chk {self.checkpoint} -step_size {self.step_size}"
        )
        if not self.tta:
            cmd += " --disable_tta"
        print(f"[predict] {cmd}", flush=True)
        cproc = subprocess.run(cmd, shell=True, check=True)
        print(cproc)
        print("Prediction finished")

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        self.check_model()
        print("Start processing")
        uuid = self.load_inputs()
        print("Start prediction")
        self.predict()
        print("Start output writing")
        self.write_outputs(uuid)


if __name__ == "__main__":
    print("START")
    Autopet_baseline().process()
