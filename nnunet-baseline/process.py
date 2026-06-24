import json
import os
import shutil
import subprocess
from pathlib import Path

import cc3d
import numpy as np
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
        Your algorithm goes here
        """
        print("nnUNet segmentation starting!")
        cproc = subprocess.run(
            f"nnUNetv2_predict -i {self.nii_path} -o {self.result_path} -d 998 -c 3d_fullres -f 0 --disable_tta",
            shell=True,
            check=True,
        )
        print(cproc)
        # since nnUNet_predict call is split into prediction and postprocess, a pre-mature exit code is received but
        # segmentation file not yet written. This hack ensures that all spawned subprocesses are finished before being
        # printed.
        print("Prediction finished")

    def apply_scribble_postprocess(self):
        """
        Reconcile the prediction with user scribbles:
          - any predicted connected component that contains a BACKGROUND click is removed
          - any component that contains a FOREGROUND click is kept (protected)
          - if a component is hit by both, FG protection wins
        Operates in-place on the predicted .nii.gz at self.result_path/self.nii_seg_file.
        Clicks are in (z, y, x) voxel index order matching the SimpleITK array layout.
        """
        seg_path = os.path.join(self.result_path, self.nii_seg_file)
        if not os.path.exists(seg_path):
            print("[postproc] seg file missing, skipping")
            return
        click_files = [f for f in os.listdir(self.lesion_click_path) if f.endswith(".json")]
        if not click_files:
            print("[postproc] no clicks file, skipping")
            return
        with open(os.path.join(self.lesion_click_path, click_files[0]), "r") as f:
            clicks = json.load(f)
        fg = clicks.get("tumor", []) or []
        bg = clicks.get("background", []) or []

        seg_img = SimpleITK.ReadImage(seg_path)
        seg = SimpleITK.GetArrayFromImage(seg_img).astype(np.uint8)
        if seg.sum() == 0:
            print("[postproc] empty prediction; nothing to do")
            return

        cc = cc3d.connected_components(seg, connectivity=26)
        n_comp = int(cc.max())
        if n_comp == 0:
            return

        def _label_at(c):
            z, y, x = int(c[0]), int(c[1]), int(c[2])
            if 0 <= z < cc.shape[0] and 0 <= y < cc.shape[1] and 0 <= x < cc.shape[2]:
                return int(cc[z, y, x])
            return 0

        kill = {lbl for lbl in (_label_at(c) for c in bg) if lbl > 0}
        protect = {lbl for lbl in (_label_at(c) for c in fg) if lbl > 0}
        kill -= protect  # contradiction → trust the FG signal

        if not kill:
            print(f"[postproc] no components removed (fg_clicks={len(fg)}, bg_clicks={len(bg)}, components={n_comp}, protected={len(protect)})")
            return

        mask_kill = np.isin(cc, list(kill))
        seg[mask_kill] = 0
        out = SimpleITK.GetImageFromArray(seg)
        out.CopyInformation(seg_img)
        SimpleITK.WriteImage(out, seg_path, True)
        print(f"[postproc] removed {len(kill)}/{n_comp} component(s) hit by bg clicks; protected={len(protect)}")

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
        print("Start scribble-aware post-processing")
        self.apply_scribble_postprocess()
        print("Start output writing")
        self.write_outputs(uuid)


if __name__ == "__main__":
    print("START")
    Autopet_baseline().process()
