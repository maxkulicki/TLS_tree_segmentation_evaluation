# TLS Tree Segmentation Evaluation

This repository provides the evaluation code for the **TreeScanPL10K**
benchmark of individual-tree segmentation from terrestrial laser scanning
data. It computes segmentation metrics, Missed/Split/Merged error frequencies,
and performance analyses across forest conditions.

## Installation

```bash
git clone https://github.com/maxkulicki/TLS_tree_segmentation_evaluation.git
cd TLS_tree_segmentation_evaluation
python -m pip install .
```

The package requires Python 3.9 or later.

## Data

TreeScanPL10K contains 272 plots from Central European forests and 10,417
manually segmented trees. The point clouds are distributed at a voxel size of
2 cm.

> Download link and DOI: *to be added on release.*

Each point cloud contains:

| Field | Description |
|---|---|
| `treeID` | Reference tree instance; 0 denotes unlabeled points |
| `completelyInside` | Indicates whether the tree is fully within the plot boundary |

## Prepare predictions

Run each method on the distributed point clouds at their native 2 cm
resolution. Evaluation uses a 10 cm voxel grid, matching the paper.

Prediction files must contain a `predID` field aligned with the reference
points. If a method filters, downsamples, or reorders the point cloud, use
`tlseval transfer` to map its predictions to the reference points:

```bash
tlseval transfer prediction.laz reference.laz -o merged.laz --dry-run
tlseval transfer prediction.laz reference.laz -o merged.laz
```

The dry run reports the proportion of reference points assigned a predicted
label and the nearest-neighbor distances used for the transfer.

## Evaluate

Evaluate one plot:

```bash
tlseval score merged.laz
```

Evaluate a directory of prediction files that already contain `treeID`:

```bash
tlseval batch predictions/ --out results/ -j 8
```

If predictions and references are stored separately, files are matched by
name:

```bash
tlseval batch predictions/ --reference references/ --out results/ -j 8
```

The batch command produces `per_tree.csv`, `per_plot.csv`, and `summary.csv`.
Errors are recorded in `failures.csv`, and processing continues with the
remaining plots.

Use the default 10 cm voxel size when comparing results with the paper. Output
files record the evaluation settings, and the following command checks whether
two result files use compatible settings:

```bash
tlseval check results_a.csv results_b.csv
```

## Generate a report

```bash
tlseval report results/ \
  --attributes data/treescanpl_plot_attributes.csv \
  --published data/treescanpl_published_results.csv \
  --out report/
```

The report includes:

- summary metrics;
- accuracy by tree-size class;
- Missed, Split, and Merged error frequencies;
- associations between accuracy and forest attributes;
- results by canopy structure and species diversity; and
- paired comparisons with the six published methods.

The attributes and published-results files are included in this repository.
If prediction filenames contain a method-specific suffix, remove it during
matching with `--strip-suffix`.

## Published results

The paper evaluates six methods on 271 of the 272 plots.

| Method | Type | Mean IoU | Detection | Precision | Recall |
|---|---|---|---|---|---|
| ForestFormer3D | Transformer | 0.757 | 0.860 | 0.839 | 0.895 |
| SegmentAnyTree | Grouping | 0.706 | 0.807 | 0.808 | 0.884 |
| TreeAIBox | Grouping | 0.697 | 0.807 | 0.790 | 0.891 |
| TreeLearn | Grouping | 0.675 | 0.776 | 0.788 | 0.875 |
| RayExtract | Algorithmic | 0.621 | 0.711 | 0.792 | 0.754 |
| treeX | Algorithmic | 0.601 | 0.688 | 0.740 | 0.860 |

Mean IoU is calculated over all evaluated trees, with unmatched trees assigned
an IoU of zero. All values use the default 10 cm evaluation grid. Per-plot
results are available in `data/treescanpl_published_results.csv` for paired
comparison with new methods.

## Metrics

Reference and predicted instances are matched one-to-one by maximizing IoU
with the Hungarian algorithm.

- `mean_iou_all` averages IoU over all reference trees, assigning zero to
  unmatched trees. This is the primary metric reported in the paper.
- `mean_iou_matched` averages IoU over reference trees assigned a prediction.
- Detection rate is the proportion of reference trees matched at IoU >= 0.5.
- Precision and recall are calculated over matched pairs on the same voxel sets
  used for IoU.

Trees that are not fully within the plot boundary are excluded when the input
contains `completelyInside` or `completely_inside`. Use `--all-trees` to include
them.

### Failure categories

Let `c(i, j)` denote the proportion of points from reference tree *i* assigned
to prediction *j*, where `j = 0` denotes background. The default dominance
threshold is 0.5, and the fragment threshold is 0.1.

| Event | Definition |
|---|---|
| **Missed** | More than half of the tree's points are assigned to background |
| **Split** | A fragment containing more than 10% of a tree is assigned to a prediction that is dominant for no reference tree |
| **Merged** | A tree's dominant prediction is also dominant for a taller tree |

The categories are independent, so a tree may have more than one flag. Their
frequencies are reported per 100 reference trees.

## Included data

| File | Contents |
|---|---|
| `data/treescanpl_plot_attributes.csv` | Plot-level forest and structural attributes |
| `data/treescanpl_published_results.csv` | Per-plot results for the six published methods |

Point-level predictions from the six published methods are not distributed.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

## Citation

```text
TODO
```

## License

See [LICENSE](LICENSE).
