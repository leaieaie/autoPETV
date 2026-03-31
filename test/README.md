# Example input PET/CT data + scribbles for the baseline model

## Input Grand Challenge Interface (`input/`)

```
input/
├── images/
│ ├── ct/
│ │ └── case_<ID>0000.mha # CT image
│ └── pet/
│ └── case<ID>_0000.mha # PET image
│
└── lesion-clicks.json # User interaction (scribbles)
```
- Grand Challenge input interface 
- **CT / PET images** are provided in `.mha` format
- Filenames must match between modalities (`case_<ID>`)
- `lesion-clicks.json` stores scribble coordinates:
  - foreground clicks (`tumor`)
  - background clicks (`background`)

---

## Raw Dataset (`images/`, `labels/`)

```
images/
├── case_<ID>0000.nii.gz # CT image
└── case<ID>_0001.nii.gz # PET image

labels/
└── case_<ID>.nii.gz # Ground-truth segmentation
```
- autoPET dataset format
- Follows **nnU-Net naming convention**
  - `_0000` → CT
  - `_0001` → PET
- Labels are binary segmentation masks

---

## Output Grand Challenge Interface (`output/`)

```
output/
└── images/
    └── tumor-lesion-segmentation/
        └── case_<ID>.mha # Predicted segmentation
```
- Grand Challenge output interface 
- Predictions are written per case
- Format: `.mha`
- Must match input case naming

---

## Final Results (`final_output/`)

```
final_output/
├── case_<ID>/
│ ├── iter_0.mha # Prediction (iteration 0)
│ ├── iter_0_scribbles.json # Scribbles used
│ ├── iter_1.mha
│ ├── iter_1_scribbles.json
│ └── ...
│
├── dice_scores.json # Dice per iteration & case
├── dice_scores_AUC.json # AUC over interactions
└── error_log.txt # Runtime errors
```

- Organized **per case** for clarity
- Each iteration contains:
  - Segmentation output
  - Corresponding interaction state
- Enables full **reproducibility of the interactive process**

---



