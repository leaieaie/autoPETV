# DEEP-PSMA → nnU-Net Conversion

This script converts the DEEP-PSMA dataset into nnU-Net-compatible format.

## What this script does
The original DEEP PSMA dataset can be found here:
NIfTI: <a href="https://doi.org/10.5281/zenodo.15281784"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15281784-blue"></a>

This script does the following steps for the DEEP PSMA dataset:
- Converts all **PSMA** and **FDG** cases to the nnUNet format
- Resamples **CT → PET space** 
- Saves:
  - CT as `imagesTr/[tracer]_[case_id]_0000.nii.gz`
  - PET as `imagesTr/[tracer]_[case_id]_0001.nii.gz`
  - TTB as `labelsTr/[tracer]_[case_id].nii.gz`
## Usage

```
python convert_deep_psma_to_nnunet_format.py \
  --root /path/to/DEEP_PSMA \
  --dest /path/to/output
```

## Scribbles and Heatmaps
We also provide pre-simulated scribbles (JSON) and heatmaps (NIfTI) so you can directly use them for model training. These are in `scribbles_deep_psma.zip` and `heatmaps_deep_psma.zip` and were produced with the `simulate_scribbles.py` script. 