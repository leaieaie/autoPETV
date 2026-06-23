# Copy locally-trained ResEncM weights into nnunet-baseline/nnUNet_results so
# `docker build .` picks them up via the Dockerfile COPY directive.
# Run on PC-C BEFORE `docker build`.
#
# Defaults assume Track B layout from PC-C_RUNBOOK.md:
#   D:\autopet\nnUNet\results\Dataset998_AutoPETV\nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres\
param(
    [string]$SrcTrainerDir = "D:\autopet\nnUNet\results\Dataset998_AutoPETV\nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
    [ValidateSet("final","best")]
    [string]$Checkpoint = "final"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $SrcTrainerDir)) { throw "Trainer dir not found: $SrcTrainerDir" }
$srcFold0 = Join-Path $SrcTrainerDir "fold_0"
$srcCkpt  = Join-Path $srcFold0 "checkpoint_$Checkpoint.pth"
if (-not (Test-Path $srcCkpt))    { throw "Checkpoint not found: $srcCkpt" }
foreach ($f in @("plans.json","dataset.json")) {
    if (-not (Test-Path (Join-Path $SrcTrainerDir $f))) {
        throw "Missing trainer-level file: $f under $SrcTrainerDir"
    }
}

$DstTrainerDir = Join-Path $PSScriptRoot "nnUNet_results\Dataset998_AutoPETV\nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
$DstFold0      = Join-Path $DstTrainerDir "fold_0"
New-Item -ItemType Directory -Force $DstFold0 | Out-Null

Copy-Item (Join-Path $SrcTrainerDir "plans.json")   $DstTrainerDir -Force
Copy-Item (Join-Path $SrcTrainerDir "dataset.json") $DstTrainerDir -Force
if (Test-Path (Join-Path $SrcTrainerDir "dataset_fingerprint.json")) {
    Copy-Item (Join-Path $SrcTrainerDir "dataset_fingerprint.json") $DstTrainerDir -Force
}
Copy-Item $srcCkpt (Join-Path $DstFold0 "checkpoint_final.pth") -Force

Write-Output "Prepared ResEncM weights under:"
Write-Output "  $DstTrainerDir"
Get-ChildItem $DstTrainerDir -Recurse -File |
    Select-Object FullName, @{N='MB';E={[math]::Round($_.Length/1MB,2)}} |
    Format-Table -AutoSize
