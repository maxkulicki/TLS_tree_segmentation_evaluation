# TLS Tree Segmentation Evaluation

Evaluation and analysis for individual-tree instance segmentation from
terrestrial laser scanning, and the reference implementation for the
TreeScanPL10K benchmark.

Given a point cloud with reference (`treeID`) and predicted (`predID`) instance
labels, it computes per-tree **IoU**, **precision** and **recall** by Hungarian
matching, classifies each reference tree's failure as **Missed**, **Split** or
**Merged**, and reports which forest attributes drive the result.

```bash
pip install -e ".[report]"
tlseval score plot.laz
```

New here? Start with the [tutorial](docs/TUTORIAL.md).

---

## Commands

| | |
|---|---|
| `tlseval score plot.laz` | score one plot |
| `tlseval batch preds/ -r reference/ -j 8` | score a directory, in parallel |
| `tlseval transfer pred.laz ref.laz -o merged.laz` | move labels onto the reference points |
| `tlseval report results/ -a attributes.csv` | correlations, stratification, figures |
| `tlseval check a.csv b.csv` | are two result files comparable? |

Or as a library:

```python
from tlseval import evaluate, summarise
df = evaluate("plot.laz")          # one row per reference tree
print(summarise(df)["mean_iou"])
```

---

## Two things that will silently corrupt a comparison

**Voxel size changes IoU by more than the gap between methods.** On this dataset
a 2 cm grid scores 0.01–0.03 higher than 10 cm, enough to reorder adjacent
methods. Every run therefore stamps its settings into the first line of its
output:

```
# evaluation_config {"all_trees": false, "dominance_threshold": 0.5,
                     "fragment_threshold": 0.1, "gt_field": "treeID",
                     "pred_field": "predID", "voxel_size": 0.1}
```

`tlseval check` compares those headers and refuses mismatched files. **The
benchmark grid is 0.1 m** — inference runs on the 2 cm clouds as distributed,
and only the scoring quantises to 10 cm.

**Predictions must sit on the same points as the reference.** Most pipelines
downsample, drop ground, or reorder, and scoring that output directly builds the
two voxel sets from different points. `tlseval transfer` does the
nearest-neighbour step once, correctly; `--dry-run` reports what fraction of
reference points actually received a label before you commit to it.

A third rule follows from the same concern: **tree attributes are read from the
point cloud, never joined from a side table by tree ID.** Annotation passes
renumber trees, so a tree-ID join risks pairing one tree's score with another
tree's measurements — silently, and without affecting any value you would think
to check.

---

## Metrics

**Mean IoU** averages over *every* evaluated reference tree, unmatched trees
contributing zero. Averaging over matched trees only answers a different
question — how well *found* trees are delineated — and runs 0.02–0.08 higher
depending on how often the method fails to match at all. The two are not
interchangeable and should never share a column.

**Detection rate** is the fraction of reference trees whose match reaches
IoU ≥ 0.5, following the convention in the literature. Counting merely-matched
trees inflates this badly: Hungarian matching pairs a tree with its best
available prediction however poor.

**Precision and recall** are point-wise over matched pairs.

### Failure taxonomy

Let `c(i, j)` be the share of reference tree *i*'s points falling in prediction
*j*, with `j = 0` meaning background. A prediction is **dominant** for tree *i*
when `c(i, j) > T`. With dominance threshold `T` (0.5) and fragment threshold `S`
(0.1):

| Event | Condition | Reading |
|---|---|---|
| **Missed** | `c(i, 0) > T` | most of the tree was left as background |
| **Split** | some `j` with `c(i, j) > S` is dominant for no tree | a fragment leaked into a prediction representing no real tree |
| **Merged** | the tree's dominant prediction also dominates a *taller* tree | the tree was absorbed into a larger neighbour |

Flags are independent, so one tree may carry more than one. Shares are computed
on **points**, not voxels, so unlike IoU the taxonomy does not move with voxel
size.

This is what separates methods that reach the same IoU by opposite routes. A
method that fragments crowns and one that fuses them can score alike while
failing an inventory in opposite directions.

### Boundary trees

If the cloud carries `completelyInside` or `completely_inside`, only flagged
trees are scored. Trees clipped by the plot edge are incomplete in the reference
itself, so scoring them penalises a method for points that were never in the
file. `--all-trees` overrides this.

---

## Input and output

**Input** — one LAS/LAZ file per plot:

| Field | Type | Description |
|---|---|---|
| `treeID` | integer | reference instance label (0 = unlabelled) |
| `predID` | integer | predicted instance label (0 = unlabelled) |
| `completelyInside` | integer | optional; 1 = tree fully within the plot |

**Output** — CSV with one row per evaluated tree: `treeID`, `matched_predID`,
`iou`, `precision`, `recall`, `gt_voxel_count`, `pred_voxel_count`, `missed`,
`split`, `merged`. `tlseval batch` additionally writes per-plot and whole-run
tables.

```python
from tlseval import read_results
df, config = read_results("results.csv")     # config comes back with the data
```

---

## Data and leaderboard

The benchmark dataset — 272 plots of Central European forest, manually
segmented, voxelised at 2 cm — is published separately; see
[`LEADERBOARD.md`](LEADERBOARD.md) for the link, current standings, and how to
add a method.

Predictions from the six methods in the paper are not distributed. This
repository is for running the evaluation yourself, on your own method, against
the same reference data and the same metric.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests use synthetic fixtures whose correct answers are derived by hand in
the comments, rather than pinned to whatever the code currently returns.

## Requirements

Python ≥ 3.9, `numpy`, `pandas`, `laspy[lazrs]`, `scipy`. `matplotlib` for
report figures.

## Citation

```
TODO
```

## License

See [LICENSE](LICENSE).
