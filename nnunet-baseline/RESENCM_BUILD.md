# Build & submit the ResEncM-weighted container

Switch this fork to the `resencm-weights` branch when you want to ship
the locally-trained ResEncM model instead of the upstream baseline.

## Prereqs
- Track B training has finished, producing
  `D:\autopet\nnUNet\results\Dataset998_AutoPETV\nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres\fold_0\checkpoint_final.pth`
  plus `plans.json` / `dataset.json` at the trainer-level folder.
- Docker Desktop running on PC-C.

## Steps (PowerShell, PC-C)
```powershell
# 0) checkout the branch (first time only)
cd $HOME\autoPETV
git fetch origin
git checkout resencm-weights
git reset --hard origin/resencm-weights

# 1) drop ResEncM weights next to the baseline weights so the Dockerfile picks them up
powershell -ExecutionPolicy Bypass -File nnunet-baseline\prepare_resencm_weights.ps1

# 2) build the container (single-platform, no attestation -> GC-friendly)
docker build --provenance=false --sbom=false --platform linux/amd64 `
  -t autopet_resencm .\nnunet-baseline

# 3) export the gzipped tar for upload
cd nnunet-baseline
Remove-Item nnunet_resencm.tar*  -ErrorAction SilentlyContinue
docker save autopet_resencm -o nnunet_resencm.tar
& "C:\Program Files\Git\usr\bin\gzip.exe" -f nnunet_resencm.tar
Get-Item nnunet_resencm.tar.gz | Select-Object Name, @{N='GB';E={[math]::Round($_.Length/1GB,2)}}
```

## Submit
Upload `nnunet-baseline\nnunet_resencm.tar.gz` to the GC algorithm
"nnUNet 3d fullres interactive (CT/PET + scribbles)" as a new container
image, wait for *Active*, then create a preliminary submission.

## Why this layout?
- `process.py` here calls `nnUNetv2_predict ... -tr nnUNetTrainer -p nnUNetResEncUNetMPlans`
  so it looks for weights under
  `nnUNet_results/Dataset998_AutoPETV/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/`.
- We do **not** commit the 800 MB checkpoint (LFS budget exhausted). The
  helper script copies it locally just before `docker build`.
