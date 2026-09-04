# Tree Instance Segmentation Evaluation

Voxel-based evaluation of tree instance segmentation from point clouds.

Given a point cloud with ground truth (`treeID`) and predicted (`predID`) instance labels, the script computes per-tree **IoU**, **precision**, and **recall** using Hungarian matching, and classifies each reference tree's failure mode as **Missed**, **Split**, or **Merged**.

## Method

1. **Voxelization** — Points are discretized into a regular 3D grid (default 0.1 m). Each voxel is encoded as a single 64-bit integer for fast set operations.
2. **Hungarian matching** — Ground truth and predicted instances are matched one-to-one by maximizing IoU using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).
3. **Per-tree metrics** — For each matched pair, intersection-over-union, precision, and recall are computed over voxel sets. Unmatched ground truth trees receive scores of zero.
4. **Failure taxonomy** — Each reference tree is additionally classified by *how* it went wrong, independently of the IoU value.

### Failure taxonomy

Let `c(i, j)` be the share of reference tree *i*'s points falling in prediction *j*, with `j = 0` denoting background. A prediction is **dominant** for tree *i* when `c(i, j) > T`. With a dominance threshold `T` (default 0.5) and a fragment threshold `S` (default 0.1):

| Event | Condition | Reading |
|---|---|---|
| **Missed** | `c(i, 0) > T` | Most of the tree was left as background |
| **Split** | some `j` with `c(i, j) > S` is dominant for no tree | A fragment leaked into a prediction representing no real tree |
| **Merged** | the tree's dominant prediction is also dominant for a *taller* tree | The tree was absorbed into a larger neighbour |

The three flags are computed independently, so one tree may carry more than one. Shares are computed on **points**, not voxels, so unlike IoU the taxonomy is unaffected by voxel size.

These events distinguish methods that reach the same IoU by different routes: a method that fragments trees and one that fuses them can score alike while failing an inventory in opposite directions.

### `completely_inside` filter

If the input file contains a `completely_inside` field (binary, 1 = tree fully within plot boundary), only those trees are evaluated by default. This avoids penalizing methods for partial trees at plot edges. Use `--all-trees` to override.

## Comparing results

**Voxel size changes IoU by more than the gap between methods.** On TreeScanPL10K, the same predictions scored 0.01–0.03 higher on a 2 cm grid than on a 10 cm grid — enough to reorder adjacent methods. Numbers produced at different voxel sizes are not comparable, and nothing about the output makes that visible on its own.

Every run therefore writes its settings into the first line of the output CSV:

```
# evaluation_config {"all_trees": false, "dominance_threshold": 0.5, "fragment_threshold": 0.1,
                     "gt_field": "treeID", "pred_field": "predID", "voxel_size": 0.1}
```

Before comparing two result files, check that these headers match. `read_results()` returns the parsed config alongside the DataFrame:

```python
from evaluate import read_results
df, config = read_results("results.csv")
```

A second rule follows from the same concern: **tree attributes are read from the point cloud, never joined from a side table by tree ID.** Annotation passes renumber trees, so joining a separate attribute table risks pairing one tree's score with another tree's measurements — silently, and without affecting any value you would think to check.

## Requirements

```
numpy
pandas
laspy
scipy
```

## Usage

```bash
# Basic evaluation
python evaluate.py plot.laz

# Custom voxel size
python evaluate.py plot.laz --voxel-size 0.05

# Evaluate all trees (ignore boundary filter)
python evaluate.py plot.laz --all-trees

# Custom field names and output path
python evaluate.py plot.laz --gt-field treeID --pred-field my_pred --output results.csv

# Adjust the failure-taxonomy thresholds
python evaluate.py plot.laz --dominance-threshold 0.5 --fragment-threshold 0.1
```

Example output:

```
Voxel size:       0.02 m
Trees evaluated:  50
Pred. instances:  59
Detection rate:   26/50 (0.520)   [IoU >= 0.5]
Matched (any IoU):43/50 (0.860)
Mean IoU:         0.480
Mean Precision:   0.565
Mean Recall:      0.704

Failure events (per 100 reference trees):
  Missed:         2.0
  Split:          14.0
  Merged:         18.0
```

## Evaluating your own method

1. Run your segmentation on the reference cloud.
2. Write the predicted instance labels back onto **the same points**, as a new field.
3. Run `evaluate.py` with `--pred-field <your field>` and the same `--voxel-size` used for the results you are comparing against.

The prediction must be point-for-point aligned with the ground truth cloud. If your pipeline resamples or reorders points, transfer the labels back by nearest neighbour before evaluating — otherwise the voxel sets are built from different points and every metric is affected.

## Input format

A single LAS/LAZ file with the following point attributes:

| Field | Type | Description |
|---|---|---|
| `treeID` | integer | Ground truth instance label (0 = unlabeled) |
| `predID` | integer | Predicted instance label (0 = unlabeled) |
| `completely_inside` | integer (optional) | 1 if tree is fully within the plot boundary |

If your predictions are in a separate file, merge them into the ground truth file first (e.g. by aligning coordinates and adding the `predID` field).

## Output

A CSV with one row per evaluated tree:

| Column | Description |
|---|---|
| `treeID` | Ground truth tree ID |
| `matched_predID` | Matched prediction ID (-1 if unmatched) |
| `iou` | Intersection over Union |
| `precision` | Fraction of predicted voxels that overlap GT |
| `recall` | Fraction of GT voxels that overlap prediction |
| `gt_voxel_count` | Number of unique GT voxels |
| `pred_voxel_count` | Number of unique predicted voxels |
| `missed` | 1 if most of the tree was left as background |
| `split` | 1 if a fragment leaked into a prediction representing no tree |
| `merged` | 1 if the tree was absorbed into a taller neighbour |

The first line of the file is the `# evaluation_config` header described under [Comparing results](#comparing-results). `pandas.read_csv(path, skiprows=1)` reads the table directly, or use `read_results()` to get the config with it.

The script also prints summary statistics (mean IoU, precision, recall, detection rate, and failure-event rates) to stdout.

## Citation

If you use this evaluation in your work, please cite:

```
TODO
```
