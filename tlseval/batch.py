"""Score many plots in one call.

Two input shapes are supported:

  merged     one file per plot carrying both treeID and predID
  pair       predictions in one directory, reference clouds in another,
             matched by filename. Labels are transferred onto the reference
             points before scoring (see transfer.py).

Output is three tables, all stamped with the same config header:

  <out>/per_tree.csv    one row per reference tree, every plot
  <out>/per_plot.csv    one row per plot
  <out>/summary.csv     one row, the whole run

Plots that fail are recorded in <out>/failures.csv rather than aborting the
run: a 272-plot job should not die on plot 3.
"""

import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .core import DEFAULTS, config_from, evaluate, summarise, write_results

SUFFIXES = (".laz", ".las")


def find_clouds(directory) -> list:
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"not a directory: {d}")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in SUFFIXES)


def _match_reference(pred_path: Path, reference_dir: Path):
    """Find the reference cloud for a prediction file, by stem then by prefix."""
    for suf in SUFFIXES:
        cand = reference_dir / (pred_path.stem + suf)
        if cand.exists():
            return cand
    stem = pred_path.stem
    for cand in sorted(reference_dir.iterdir()):
        if cand.suffix.lower() in SUFFIXES and stem.startswith(cand.stem):
            return cand
    return None


def _score_one(args) -> dict:
    """Worker. Returns a record; never raises, so one bad plot cannot kill a run."""
    pred_path, ref_path, opts = args
    plot = Path(pred_path).stem
    try:
        if ref_path is None:
            path = pred_path
            tmp = None
        else:
            from .transfer import transfer_labels
            tmp = transfer_labels(pred_path, ref_path, pred_field=opts["pred_field"],
                                  tolerance=opts["tolerance"])
            path = tmp
        df = evaluate(
            str(path),
            voxel_size=opts["voxel_size"],
            gt_field=opts["gt_field"],
            pred_field=opts["pred_field"],
            all_trees=opts["all_trees"],
            dominance_threshold=opts["dominance_threshold"],
            fragment_threshold=opts["fragment_threshold"],
        )
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        df.insert(0, "plot", plot)
        return {"plot": plot, "ok": True, "per_tree": df,
                "summary": {"plot": plot, **summarise(df)}}
    except (Exception, SystemExit) as exc:
        return {"plot": plot, "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


def run(
    predictions,
    reference=None,
    out_dir="results",
    jobs=1,
    tolerance=0.05,
    progress=True,
    **cfg,
):
    """Score every cloud in `predictions`, writing three tables into `out_dir`.

    predictions  directory of clouds, or a list of paths
    reference    directory of reference clouds; None if predictions already
                 carry treeID (the "merged" shape)
    jobs         worker processes; each holds one plot in memory
    """
    opts = {k: cfg.get(k, v) for k, v in DEFAULTS.items()}
    opts["tolerance"] = tolerance
    config = config_from(**{k: opts[k] for k in DEFAULTS})

    paths = ([Path(p) for p in predictions] if not isinstance(predictions, (str, Path))
             else find_clouds(predictions))
    if not paths:
        raise SystemExit(f"error: no .laz/.las files found in {predictions}")

    ref_dir = Path(reference) if reference else None
    tasks = []
    missing = []
    for p in paths:
        ref = _match_reference(p, ref_dir) if ref_dir else None
        if ref_dir and ref is None:
            missing.append(p.name)
            continue
        tasks.append((str(p), str(ref) if ref else None, opts))
    if missing:
        print(f"warning: no reference cloud for {len(missing)} file(s): "
              f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}",
              file=sys.stderr)
    if not tasks:
        raise SystemExit("error: nothing to score")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records, done = [], 0
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_score_one, t) for t in tasks]
            for fut in as_completed(futures):
                records.append(fut.result())
                done += 1
                if progress:
                    _tick(done, len(tasks), records[-1])
    else:
        for t in tasks:
            records.append(_score_one(t))
            done += 1
            if progress:
                _tick(done, len(tasks), records[-1])

    ok = [r for r in records if r["ok"]]
    bad = [r for r in records if not r["ok"]]

    if not ok:
        raise SystemExit("error: every plot failed; see failures.csv")

    per_tree = pd.concat([r["per_tree"] for r in ok], ignore_index=True)
    per_plot = pd.DataFrame([r["summary"] for r in ok]).sort_values("plot")

    write_results(per_tree, out / "per_tree.csv", config)
    write_results(per_plot, out / "per_plot.csv", config)

    overall = pd.DataFrame([{
        "n_plots": len(per_plot),
        "n_trees": len(per_tree),
        # Plot-level means: each plot weighs the same regardless of tree count,
        # which is what the paper reports. Tree-level means are also given.
        "mean_iou_plotmean": per_plot["mean_iou"].mean(),
        "mean_iou_treemean": per_tree["iou"].mean(),
        "detection_rate": per_plot["detection_rate"].mean(),
        "mean_precision": per_plot["mean_precision"].mean(),
        "mean_recall": per_plot["mean_recall"].mean(),
        "missed_per100": 100 * per_tree["missed"].mean(),
        "split_per100": 100 * per_tree["split"].mean(),
        "merged_per100": 100 * per_tree["merged"].mean(),
        "n_failed_plots": len(bad),
    }])
    write_results(overall, out / "summary.csv", config)

    if bad:
        pd.DataFrame([{k: r[k] for k in ("plot", "error", "traceback")} for r in bad]) \
          .to_csv(out / "failures.csv", index=False)

    if progress:
        print()
        _print_summary(overall.iloc[0], bad, out, config)
    return per_tree, per_plot, overall


def _tick(done, total, rec):
    mark = "." if rec["ok"] else "!"
    end = "" if done % 50 else f" {done}/{total}\n"
    print(mark, end=end, flush=True)


def _print_summary(row, bad, out, config):
    print(f"voxel size      {config['voxel_size']} m")
    print(f"plots scored    {int(row['n_plots'])}"
          + (f"   ({len(bad)} failed)" if bad else ""))
    print(f"trees scored    {int(row['n_trees'])}")
    print(f"mean IoU        {row['mean_iou_plotmean']:.3f}  (plot mean)")
    print(f"                {row['mean_iou_treemean']:.3f}  (tree mean)")
    print(f"detection rate  {row['detection_rate']:.3f}")
    print(f"precision       {row['mean_precision']:.3f}")
    print(f"recall          {row['mean_recall']:.3f}")
    print("failures per 100 reference trees:")
    print(f"  missed {row['missed_per100']:.1f}   split {row['split_per100']:.1f}"
          f"   merged {row['merged_per100']:.1f}")
    print(f"\nwritten to {out}/  (per_tree.csv, per_plot.csv, summary.csv"
          + (", failures.csv)" if bad else ")"))
    if bad:
        print(f"\n{len(bad)} plot(s) failed:")
        for r in bad[:5]:
            print(f"  {r['plot']}: {r['error']}")
        if len(bad) > 5:
            print(f"  ... and {len(bad) - 5} more, see {out}/failures.csv")
