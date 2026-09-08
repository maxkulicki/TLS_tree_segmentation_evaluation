# Tutorial

From a fresh clone to a scored, analysed result. The first section needs no
download and takes under a minute.

---

## 1. Install and prove it works

```bash
git clone https://github.com/maxkulicki/TLS_tree_segmentation_evaluation
cd TLS_tree_segmentation_evaluation
pip install -e ".[report]"
```

`[report]` adds matplotlib for the figures. Without it everything still runs and
the tables are still written; only the plots are skipped.

Then:

```bash
tlseval demo
```

This generates a synthetic plot, runs a small built-in baseline segmenter on it,
and scores the result:

```
synthetic plot -> demo/plot.laz
  79,002 points, 24 reference trees (18 fully inside the plot)
  naive baseline produced 19 instances

voxel size        0.1 m
trees evaluated   18   (6 boundary-clipped excluded via 'completelyInside')
pred. instances   19
mean IoU          0.507
detection rate    0.556   [IoU >= 0.5]
mean precision    0.562
mean recall       0.640

failure events per 100 reference trees:
  missed 0.0   split 0.0   merged 27.8
```

Read that last line before moving on. The baseline finds stems in a slice above
the ground, so it never *loses* a tree (Missed 0.0) and never *fragments* one
(Split 0.0) — but it fuses understorey trees into the dominant crown above them
(Merged 27.8). Two methods can reach the same mean IoU by opposite routes, and
the taxonomy is the only place that difference shows up.

The baseline lives in `tlseval/demo.py` and is about 40 lines. It is a reasonable
starting point if you want to see the shape of a segmenter that this tool can
score.

---

## 2. Score one of your own plots

The scorer reads **one file** containing both labels on **the same points**:

| Field | Meaning |
|---|---|
| `treeID` | reference instance, 0 = unlabelled |
| `predID` | your prediction, 0 = unlabelled |
| `completelyInside` | optional, 1 = tree fully inside the plot |

```bash
tlseval score my_plot.laz
```

If your fields are named differently:

```bash
tlseval score my_plot.laz --gt-field tree_id --pred-field my_labels
```

### If your prediction is a separate file

It usually is, and it usually does not sit on the same points — most pipelines
downsample, drop ground, or reorder. Scoring that directly builds the two voxel
sets from different points, and every number is wrong without anything looking
wrong.

Check the damage first:

```bash
tlseval transfer my_prediction.laz reference.laz -o merged.laz --dry-run
```

```
reference points   4,035,284
predicted points   2,410,882
predicted instances      71
assigned fraction  0.588   (tolerance 0.05 m)
nn distance        median 0.021 m, p95 0.112 m
```

`assigned fraction` is the number to watch. Here 59% of reference points found a
predicted label within 5 cm — plausible for a method that discards ground and
understorey. If it comes out near zero, the tolerance is too tight for your
prediction's resolution: compare it against the p95 distance and raise it.

Then write the merged file and score it:

```bash
tlseval transfer my_prediction.laz reference.laz -o merged.laz
tlseval score merged.laz
```

---

## 3. Run the whole benchmark

Download the dataset, then:

```bash
# predictions already carry treeID
tlseval batch predictions/ --out results/ -j 8

# predictions are separate files, matched to reference clouds by filename
tlseval batch predictions/ --reference reference/ --out results/ -j 8
```

```
..................................................  50/272
..................................................  100/272
...

voxel size      0.1 m
plots scored    272
trees scored    9485
mean IoU        0.757  (plot mean)
                0.756  (tree mean)
detection rate  0.861
precision       0.816
recall          0.871
failures per 100 reference trees:
  missed 2.8   split 12.1   merged 4.4

written to results/  (per_tree.csv, per_plot.csv, summary.csv)
```

Three tables come out:

- `per_tree.csv` — one row per reference tree, every plot
- `per_plot.csv` — one row per plot
- `summary.csv` — one row for the run

A plot that fails is recorded in `failures.csv` with its traceback and the run
carries on. A 272-plot job should not die on plot 3.

### One thing that will bite you

**Voxel size changes IoU by more than the gap between methods.** On this dataset
a 2 cm grid scores 0.01–0.03 higher than 10 cm — larger than the difference
between adjacent published methods. Nothing about a results file makes that
visible on its own, so every run writes its settings into the first line:

```
# evaluation_config {"all_trees": false, "dominance_threshold": 0.5, ...
```

Before comparing two result files, check they match:

```bash
tlseval check results_mine/per_plot.csv results_theirs/per_plot.csv
```

```
MISMATCH: these files were produced under different settings and are not comparable.
  voxel_size: 0.02, 0.1
```

**The benchmark grid is 0.1 m.** Inference runs on the 2 cm clouds as
distributed; only the scoring quantises to 10 cm. Leave `--voxel-size` alone
unless you know why you are changing it.

---

## 4. Analyse the result

Scoring tells you how well a method did. The report tells you *where* and *why*.

```bash
tlseval report results/ --attributes plot_attributes.csv --out report/
```

If your prediction files carry a method suffix (`Plot_A_mymethod.laz`), the join
to plot attributes will fail and tell you exactly what to add:

```
  'Rem_Gorlice_2015_0101703_tls2trees' starts with the plot name 'Rem_Gorlice_2015_0101703'.
  Retry with:  --strip-suffix '_tls2trees'
```

The report writes `summary.md`, a set of CSVs and four figures:

- **attribute sensitivity** — Spearman ρ of every plot attribute against mean
  IoU. On the published benchmark, Mean CAI (how many crown layers stack above
  each ground cell) leads at ρ̄ ≈ −0.58 and is the top-ranked predictor for every
  method individually.
- **size breakdown** — accuracy by reference-tree size. Errors concentrate in the
  small classes, which is where an inventory can least absorb them.
- **median splits and a 2×2 stratification** — performance under contrasting
  forest conditions.
- **failure profile** — Missed / Split / Merged per 100 reference trees.

Attributes are joined **by plot name, never by tree ID**. Annotation passes
renumber trees, and a tree-ID join silently pairs one tree's score with another
tree's measurements — without changing any value you would think to check.

---

## 5. Submit to the leaderboard

See [`../LEADERBOARD.md`](../LEADERBOARD.md). In short: run `tlseval batch` at the
default voxel size, open a pull request adding your row and the `summary.csv`
that produced it, and state whether your method saw any of this data in training.

---

## Reference: what the metrics mean

**IoU** is computed on voxel sets, per matched pair. Instances are matched
one-to-one by the Hungarian algorithm, maximising IoU.

**Mean IoU** averages over *every* evaluated reference tree, with unmatched trees
contributing zero. Averaging over matched trees only answers a different question
— "how well are found trees delineated" — and inflates the result by 0.02–0.08
depending on how often the method fails to match at all. The two are not
interchangeable.

**Detection rate** is the fraction of reference trees whose match reaches
IoU ≥ 0.5. Counting merely-matched trees instead inflates it badly: Hungarian
matching pairs a tree with its best available prediction however poor that is.

**Precision and recall** are point-wise, over matched pairs.

**Missed / Split / Merged** are computed on point shares rather than voxels, so
unlike IoU they do not move with voxel size:

| Event | Condition | Reading |
|---|---|---|
| Missed | most of the tree was left as background | the tree was not found |
| Split | a fragment ≥ 10% leaked into a prediction representing no tree | the tree was fragmented |
| Merged | the tree's dominant prediction also dominates a *taller* tree | the tree was absorbed |

The flags are independent; one tree may carry more than one.

**Boundary trees.** If the cloud carries `completelyInside` (or
`completely_inside`), only flagged trees are scored. Trees clipped by the plot
edge are incomplete in the reference itself, so scoring them penalises a method
for points that were never in the file. `--all-trees` overrides this.
