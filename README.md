# autoPETV
Official repository for autoPET V machine learning challenge

---



## Scribble Simulation

All the scribble simulation logic is implemented in `simulate_scribbles.py`

Given a binary volume `L`, we simulate scribbles using the following procedure:

1. Identify the slice with the largest foreground area in `L`
2. Generate a scribble within this slice using one of three strategies:
   - `centerline`
   - `random`
   - `boundary`
3. Export the result either as:
   - a `JSON` file containing scribble coordinates, or
   - a `NIfTI` file with the same spatial dimensions as the PET/CT images containing Gaussian Heatmaps of the scribbles

---

### How to simulate a scribble from a binary NumPy array
If you want to simulate scribbels from a binary array, for example the grount-truth label or from the model's error, you can use our simulation functions as follows:

```python
from simulate_scribbles import simulate_scribble_from_label, generate_heatmap_from_scribbles

your_binary_numpy_array = some_function(...)
strategy = "centerline"  # alternatives: "random", "boundary"
sigma = 0  # Gaussian heatmap radius

# Generate scribble coordinates and class label
scribbles, label_cls = simulate_scribble_from_label(
    your_binary_numpy_array,
    strategy
)

# Optionally convert scribbles to a Gaussian heatmap volume
heatmap_volume = generate_heatmap_from_scribbles(
    scribbles,
    sigma=0
)
```
You can then use either `scribbles` (3D coordinates) or `heatmap_volume` (3D volume) in your models. For more details on the implementation, see `simulate_scribbles.py`

### How scribbles will be simulated during testing

We use the script `interactive_loop.py` to simulate scribbles during the test phase. Three types of scribbles: **centerline**, **random**, and **boundary**, will be distributed uniformly across all test cases (each type applied to one-third of the cases, consistent for all participants).

Each test case will be evaluated over **6 interactive steps**: one initial prediction followed by five corrective steps.

1. **Step 1:** Prediction **without scribbles**
2. **Steps 2–6:** Iterative correction using simulated scribbles based on the model’s largest error region:
   - If the largest error is **over-segmentation** → apply a **background scribble**
   - If the largest error is **under-segmentation** → apply a **foreground scribble**
   - Then generate a new prediction **with the updated scribbles**


## Infer with the Baseline nnUNet Model
Make sure you have `git lfs` installed and have downloaded all large files from the repository. 
 
```python
python interactive_loop.py --input_cases test/ --input_interface test/ --result_dir test/final_output/ --strategy random
```