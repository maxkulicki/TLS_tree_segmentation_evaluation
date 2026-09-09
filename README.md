# TLS Tree Segmentation Evaluation

Evaluation and analysis for individual-tree instance segmentation from
terrestrial laser scanning, and the reference implementation for the
**TreeScanPL10K** benchmark.

Run your segmentation method on our data and get back the same numbers and the
same analyses reported in the paper: per-tree IoU, precision and recall by
Hungarian matching, a Missed / Split / Merged failure profile, and the
forest-structure analysis showing which plots are hard and why.

```bash
pip install -e ".[report]"

tlseval batch  predictions/ --out results/ -j 8
tlseval report results/ -a data/treescanpl_plot_attributes.csv \
                        -p data/treescanpl_published_results.csv --out report/
```

---

## 1. Install

```bash
git clone https://github.com/maxkulicki/TLS_tree_segmentation_evaluation
cd TLS_tree_segmentation_evaluation
pip install -e ".[report]"
```

`[report]` adds matplotlib for the figures. Without it everything still runs and
all tables are written; only the plots are skipped.

Python ≥ 3.9, `numpy`, `pandas`, `laspy[lazrs]`, `scipy`.

## 2. Get the data

**TreeScanPL10K** — 272 plots of Central European forest, 10,417 manually
segmented trees, voxelised at 2 cm.

> Download link and DOI: *to be added on release.*

Each cloud carries:

| Field | Meaning |
|---|---|
| `treeID` | reference instance, 0 = unlabelled |
| `completelyInside` | 1 = the tree lies fully inside the plot boundary |

## 3. Run your method

Run inference on the clouds **as distributed**, at their native 2 cm. Only the
scoring quantises, to 10 cm.

Write your instance labels back as a `predID` field. The scorer needs both
labels on *the same points*, and most pipelines do not return them that way —
they downsample, drop ground, or reorder. If yours does, use the transfer helper
rather than writing your own nearest-neighbour step:

```bash
# check coverage first
tlseval transfer my_prediction.laz reference.laz -o merged.laz --dry-run
```

```
reference points   4,035,284
predicted points   2,410,882
predicted instances      71
assigned fraction  0.588   (tolerance 0.05 m)
nn distance        median 0.021 m, p95 0.112 m
```

`assigned fraction` is what to watch: 59% of reference points found a label
within 5 cm, which is plausible for a method that discards ground and
understorey. Near zero means the tolerance is too tight for your prediction's
resolution — compare it against the p95 distance and raise `--tolerance`.

Then write it for real:

```bash
tlseval transfer my_prediction.laz reference.laz -o merged.laz
```

## 4. Score

One plot:

```bash
tlseval score merged.laz
```

```
voxel size        0.1 m
trees evaluated   45   (5 boundary-clipped excluded via 'completelyInside')
pred. instances   71
mean IoU          0.512   (43/45 matched trees)
                  0.489   (all trees, unmatched = 0)
detection rate    0.400   [IoU >= 0.5]
mean precision    0.563
mean recall       0.690

failure events per 100 reference trees:
  missed 0.0   split 15.6   merged 8.9
```

The whole benchmark:

```bash
# your files already carry treeID
tlseval batch predictions/ --out results/ -j 8

# or predictions and reference in separate directories, matched by filename
tlseval batch predictions/ --reference reference/ --out results/ -j 8
```

Three tables come out — `per_tree.csv`, `per_plot.csv`, `summary.csv`. A plot
that fails is recorded in `failures.csv` with its traceback and the run carries
on; a 272-plot job should not die on plot 3.

> **Keep the voxel size at the default.** It is the one setting that changes IoU
> by more than the gap between methods — a 2 cm grid scores 0.01–0.03 higher than
> 10 cm on this data. Every output records the settings it was made under in its
> first line, and `tlseval check a.csv b.csv` compares two files and refuses
> mismatched ones.

## 5. Analyse

This is the part that reproduces the paper's analysis for your method.

```bash
tlseval report results/ \
  --attributes data/treescanpl_plot_attributes.csv \
  --published  data/treescanpl_published_results.csv \
  --out report/
```

Both data files ship with this repository. `--attributes` is what enables the
forest-structure analysis; `--published` adds the comparison against the six
methods in the paper. Neither is required — without them the failure profile,
size breakdown and extreme-plot tables still run.

You get `summary.md`, a CSV per analysis, and figures:

- **What makes a plot hard** — Spearman ρ of every plot attribute against mean
  IoU, grouped by attribute family. On the published benchmark Mean CAI, which
  counts how many crown layers stack above each ground cell, is the strongest
  single correlate and the top-ranked predictor for every method individually.
- **Easiest vs hardest plots** — the best and worst quarter, contrasted attribute
  by attribute and standardised so the differences can be ranked.
- **By canopy structure and diversity** — a median split on Mean CAI and the
  Shannon index; every plot lands in one of four cells.
- **Failure profile** — Missed / Split / Merged per 100 reference trees.
- **Accuracy by tree size** — errors concentrate in the small classes, which is
  where an inventory can least absorb them.
- **Against the published methods** — your per-plot IoU next to all six, on the
  plots you both cover.

If your prediction files carry a method suffix (`Plot_A_mymethod.laz`), the join
to plot attributes fails and tells you what to add:

```
  'Rem_Gorlice_2015_0101703_tls2trees' starts with the plot name 'Rem_Gorlice_2015_0101703'.
  Retry with:  --strip-suffix '_tls2trees'
```

Attributes are joined **by plot name, never by tree ID**. Annotation passes
renumber trees, so a tree-ID join can pair one tree's score with another tree's
measurements — silently, and without affecting any value you would think to
check.

---

## Published results

The six methods evaluated in the paper, on all 272 plots.

| Method | Type | Mean IoU | Detection | Precision | Recall |
|---|---|---|---|---|---|
| ForestFormer3D | Transformer | 0.778 | 0.862 | 0.839 | 0.895 |
| SegmentAnyTree | Grouping | 0.744 | 0.808 | 0.809 | 0.884 |
| TreeAIBox | Grouping | 0.719 | 0.807 | 0.791 | 0.892 |
| TreeLearn | Grouping | 0.719 | 0.776 | 0.789 | 0.875 |
| treeX | Algorithmic | 0.683 | 0.722 | 0.763 | 0.871 |
| RayExtract | Algorithmic | 0.626 | 0.709 | 0.792 | 0.753 |

Per-plot values are in `data/treescanpl_published_results.csv` — the file
`tlseval report --published` compares against.

Point-level predictions from these six methods are not distributed. This
repository is for running the evaluation yourself, on your own method, against
the same data and the same metric. If you would like a result listed here, open
an issue or a pull request with your `summary.csv`.

---

## Metrics

**IoU** is computed on voxel sets. Reference and predicted instances are matched
one-to-one by the Hungarian algorithm, maximising IoU.

**Mean IoU** is reported two ways, because the choice moves the number by
0.02–0.08 and the two answer different questions:

- `mean_iou_matched` — over reference trees that got a match. *How well are
  found trees delineated?* This is the convention behind the table above, so it
  is the one that reproduces those numbers.
- `mean_iou_all` — over every evaluated tree, unmatched ones scoring zero. *How
  well is the plot segmented?* This is the convention the detection rate uses.

**Detection rate** is the fraction of reference trees whose match reaches
IoU ≥ 0.5. Counting merely-matched trees instead inflates it badly: Hungarian
matching pairs a tree with its best available prediction however poor.

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

This is what separates methods reaching the same IoU by opposite routes: one
fragmenting crowns, one fusing them, failing an inventory in opposite
directions.

### Boundary trees

If the cloud carries `completelyInside` (or `completely_inside`), only flagged
trees are scored. Trees clipped by the plot edge are incomplete in the reference
itself, so scoring them penalises a method for points that were never in the
file. `--all-trees` overrides this.

---

## Commands

| | |
|---|---|
| `tlseval score plot.laz` | score one plot |
| `tlseval batch preds/ -r reference/ -j 8` | score a directory, in parallel |
| `tlseval transfer pred.laz ref.laz -o merged.laz` | move labels onto the reference points |
| `tlseval report results/ -a … -p …` | the full analysis |
| `tlseval check a.csv b.csv` | are two result files comparable? |

As a library:

```python
from tlseval import evaluate, summarise
df = evaluate("plot.laz")               # one row per reference tree
print(summarise(df)["mean_iou_matched"])

from tlseval import read_results
df, config = read_results("results.csv")  # settings come back with the data
```

**Output columns** — `treeID`, `matched_predID`, `iou`, `precision`, `recall`,
`gt_voxel_count`, `pred_voxel_count`, `missed`, `split`, `merged`.

## What ships here

| | |
|---|---|
| `data/treescanpl_plot_attributes.csv` | 271 plots × 29 attributes: CAI, Shannon, DBH, height, Clark–Evans, species |
| `data/treescanpl_published_results.csv` | per-plot results for the six published methods |

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests use small fixtures whose expected values are derived by hand in the
comments rather than pinned to current behaviour.

## Citation

```
TODO
```

## License

See [LICENSE](LICENSE).
