# Leaderboard — TreeScanPL10K

Individual-tree instance segmentation, 272 plots of Central European forest.

**Scoring:** 10 cm voxel grid · unmatched reference trees score zero ·
boundary-clipped trees excluded via `completelyInside`.

```
# evaluation_config {"all_trees": false, "dominance_threshold": 0.5,
                     "fragment_threshold": 0.1, "gt_field": "treeID",
                     "pred_field": "predID", "voxel_size": 0.1}
```

Rows scored under any other configuration are not comparable and will not be
merged. Run `tlseval check` against a listed `summary.csv` before submitting.

> **Provisional.** These numbers are being re-derived against the corrected
> scoring convention and will be finalised alongside the paper. Do not cite this
> table yet; cite the paper.

---

## Standings

| # | Method | Type | Mean IoU | Detection | Verified | Trained on this data |
|---|---|---|---|---|---|---|
| 1 | ForestFormer3D | Transformer | 0.757 | 0.861 | ✓ | no |
| 2 | SegmentAnyTree | Grouping | 0.706 | 0.807 | ✓ | no |
| 3 | TreeAIBox | Grouping | 0.697 | 0.807 | ✓ | no |
| 4 | TreeLearn | Grouping | 0.675 | 0.776 | ✓ | no |
| 5 | RayExtract | Algorithmic | 0.628 | 0.719 | ✓ | no |
| 6 | treeX | Algorithmic | 0.601 | 0.688 | ✓ | no |

Mean IoU is the mean over plots of each plot's mean tree IoU. See the
[paper](#citation) for per-method failure profiles, runtimes and the structural
analysis.

---

## Adding your method

1. **Get the data.** Link and DOI: *(to be added on release.)* Use the clouds as
   distributed — 2 cm voxelised. Run inference at that resolution; only the
   scoring quantises to 10 cm.

2. **Produce labels on the reference points.** Write a `predID` field onto the
   reference cloud. If your pipeline resamples or reorders, use the transfer
   helper rather than writing your own:

   ```bash
   tlseval transfer my_pred.laz reference.laz -o merged.laz --dry-run   # check coverage
   tlseval transfer my_pred.laz reference.laz -o merged.laz
   ```

3. **Score the whole benchmark** at the default settings:

   ```bash
   tlseval batch merged/ --out results/ -j 8
   ```

4. **Open a pull request** adding a row to the table above and your
   `results/summary.csv` under `leaderboard/<your-method>/`. Include:

   | Field | |
   |---|---|
   | Method name and version | |
   | Citation or repository | |
   | Licence | |
   | Type | transformer / grouping / algorithmic / hybrid |
   | Hardware and runtime per plot | |
   | **Trained on TreeScanPL?** | any overlap, including pre-training |
   | **Deterministic?** | if not, report the spread over repeated runs |

The last two are required, not optional, because both silently invalidate a
comparison. A method that saw any of these plots in training is not competing on
the same terms. A method that varies run to run — unseeded sampling is the usual
cause, and can move IoU by ±0.01 — needs that spread on the record rather than a
single lucky run.

### Verified vs self-reported

A row is marked **verified** once a maintainer has re-scored the submitted
predictions and reproduced the numbers. Rows without that check are labelled
**self-reported** and are still welcome — the label is about provenance, not
quality.

---

## Notes on comparing rows

**Instance counts vary enormously between methods** and this protocol does not
penalise extra instances. That is deliberate: `predID` is a per-point label, so
predicted instances are a disjoint partition of the points — an extra instance
can only be carved out of an existing one, which lowers the IoU of whatever it
was split from. What remains unpenalised is spurious instances made of ground
and understorey points, and those are filterable downstream. A plot whose real
trees are all recovered is usable however many stray labels sit in the leftovers.

**Mean IoU alone is a poor summary.** Two methods can reach the same value by
opposite routes — one fragmenting crowns, one fusing them. The Missed / Split /
Merged profile in `tlseval report` is where that separates, and it matters more
for inventory use than the mean does.

## Citation

```
TODO
```
