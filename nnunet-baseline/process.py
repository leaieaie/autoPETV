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
        """
        Run nnUNetv2 inference with reduced (4x) test-time mirroring.

        Full 8x mirror TTA measured ~95s/predict (~146s/step) on RTX 3060, which
        risks the 15 min/case budget over 6 interactive steps (and the eval GPU
        is a slower T4). Restricting mirroring to 2 axes (4x) ~halves predict time
        while keeping most of the TTA benefit. The CLI nnUNetv2_predict cannot
        select mirror axes, so we drive the predictor API directly.
        """
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        print("nnUNet segmentation starting (4x mirror TTA)!")
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device("cuda", 0),
            verbose=False,
            allow_tqdm=True,
        )
        model_folder = os.path.join(
            os.environ["nnUNet_results"],
            "Dataset998_AutoPETV",
            "nnUNetTrainer__nnUNetPlans__3d_fullres",
        )
        predictor.initialize_from_trained_model_folder(
            model_folder, use_folds=(0,), checkpoint_name="checkpoint_final.pth"
        )
        # Keep only 2 mirror axes (4x) instead of the trained full set (8x) to fit
        # the per-case time budget. Intersect so we never enable an untrained axis.
        trained_axes = predictor.allowed_mirroring_axes
        if trained_axes:
            predictor.allowed_mirroring_axes = tuple(a for a in trained_axes if a in (0, 1))
        print(f"[tta] mirror axes: trained={trained_axes} -> used={predictor.allowed_mirroring_axes}")

        predictor.predict_from_files(
            self.nii_path,
            self.result_path,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
        print("Prediction finished")

   
    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        print("Start processing")
        uuid = self.load_inputs()
        print("Start prediction")
        self.predict()
        print("Start output writing")
        self.write_outputs(uuid)


if __name__ == "__main__":
    print("START")
    Autopet_baseline().process()
