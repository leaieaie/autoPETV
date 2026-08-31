# Final submission — LesionLocator 10-fold ensemble

Our autoPET V final submission does not use a network we trained. It is an
ensemble of the ten publicly released LesionLocator interactive folds. The
challenge rules state that publicly available data and networks may all be
used; this file records exactly what we changed so the submission can be
reproduced and so the upstream authors are properly credited.

## Upstream

| | |
|---|---|
| Repository | https://github.com/MIC-DKFZ/autoPET-interactive |
| Commit | `0da0e7f` ("docker building utils") |
| License | Apache-2.0 |
| Paper | LesionLocator, arXiv:2508.21680 |
| Weights | Released by the upstream authors via Google Drive (7.70 GB zip, 8.2 GB extracted, ten folds). We do not redistribute them. |

The upstream code already targets all ten folds
(`use_folds=(0,1,2,3,4,5,6,7,8,9)`), so no change was needed there.

## Our changes to `inference.py`

One functional change, plus two cosmetic artifacts introduced by the editor
(a UTF-8 BOM on the first line and trailing newlines at the end of the file;
neither affects execution).

```diff
@@ -59,7 +59,7 @@ def run():
     predictor = autoPETPredictor(
         tile_step_size=0.5,
         use_gaussian=True,
-        use_mirroring=True, # Set for False for faster inference on GC
+        use_mirroring=False, # Set for False for faster inference on GC
         perform_everything_on_device=True,
         device=device,
         verbose=True,
```

Test-time mirroring is disabled because the ten-fold ensemble already consumes
most of the per-case time budget. The comment recommending this is the upstream
authors' own.

## Building the model folder

`scripts/prepare_lesionlocator_model.py` (this repository) extracts the folds
from the released archive and strips each checkpoint to the four keys the
nnU-Net predictor reads — `network_weights`, `trainer_name`,
`inference_allowed_mirroring_axes`, `init_args`. Everything else in the file is
training bookkeeping (optimizer state, gradient scaler, logging) that inference
never touches. Each checkpoint is reloaded and verified after being written.

```
python scripts/prepare_lesionlocator_model.py --folds 0 1 2 3 4 5 6 7 8 9
```

Per fold: 820 MB → 391 MB. Model directory: 8.2 GB → 3.9 GB.

## Building the container

```
docker build --provenance=false --sbom=false --platform linux/amd64 -t lesionlocator-10f .
```

The three flags matter: Grand Challenge rejects images carrying provenance or
SBOM attestations with an import failure, and the platform must be pinned when
building from a non-amd64 host.

## Local verification

Measured on an RTX 3060 12 GB (2026-08-27), one PSMA case from `test/input`:

| | |
|---|---|
| Wall time, one iteration | 171.9 s (161.0 s prediction, ~12.9 s per fold, ~11 s startup) |
| Five iterations (one case) | 859.5 s against the 15-minute budget |
| Input | 371×256×256, resampled to 404×344×344, 36 tiles at patch 192³ |
| Output | shape restored to 371×256×256, 22,618 foreground voxels |
| Dice vs. ground truth | 0.9387 (GT 23,060 voxels) |

`--shm-size=8g` is required when running locally. Docker's default `/dev/shm`
of 64 MB makes nnU-Net's preprocessing workers die with no error message
(`RuntimeError: Background workers died`). This is a runtime flag only and does
not affect the image.

```
docker run --rm --gpus all --shm-size=8g -v "<repo>/test/input:/input:ro" -v "<out>:/output" lesionlocator-10f
```
