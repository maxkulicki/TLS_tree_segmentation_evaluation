# Tree Instance Segmentation Evaluation

Voxel-based evaluation of tree instance segmentation from point clouds.

Given a point cloud with ground truth (`treeID`) and predicted (`predID`) instance labels, the script computes per-tree **IoU**, **precision**, and **recall** using Hungarian matching.

## Method

1. **Voxelization** — Points are discretized into a regular 3D grid (default 0.1 m). Each voxel is encoded as a single 64-bit integer for fast set operations.
2. **Hungarian matching** — Ground truth and predicted instances are matched one-to-one by maximizing IoU using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).
3. **Per-tree metrics** — For each matched pair, intersection-over-union, precision, and recall are computed over voxel sets. Unmatched ground truth trees receive scores of zero.

### `completely_inside` filter

If the input file contains a `completely_inside` field (binary, 1 = tree fully within plot boundary), only those trees are evaluated by default. This avoids penalizing methods for partial trees at plot edges. Use `--all-trees` to override.

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
```

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

The script also prints summary statistics (mean IoU, precision, recall, detection rate) to stdout.

## Citation

If you use this evaluation in your work, please cite:

```
TODO
```
