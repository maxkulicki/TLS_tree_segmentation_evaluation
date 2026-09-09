"""Command line: score, batch, transfer, report, check."""

import argparse
import json
import sys

from . import __version__
from .core import (DEFAULTS, DETECTION_IOU_THRESHOLD, TlsEvalError, config_from,
                   evaluate, read_results, summarise, write_results)

EPILOG = """\
Results are comparable only across runs sharing a voxel size. Every output
carries the settings it was produced under; `tlseval check` compares them.
"""


def _common(p):
    p.add_argument("--voxel-size", "-v", type=float, default=DEFAULTS["voxel_size"],
                   help=f"metres (default: {DEFAULTS['voxel_size']}, the benchmark grid)")
    p.add_argument("--gt-field", default=DEFAULTS["gt_field"])
    p.add_argument("--pred-field", default=DEFAULTS["pred_field"])
    p.add_argument("--all-trees", action="store_true",
                   help="score boundary-clipped trees too (default: skip them)")
    p.add_argument("--dominance-threshold", type=float, default=DEFAULTS["dominance_threshold"],
                   help="T: share of a tree's points making a prediction dominant")
    p.add_argument("--fragment-threshold", type=float, default=DEFAULTS["fragment_threshold"],
                   help="S: share above which a leaked fragment counts as a Split")


def build_parser():
    ap = argparse.ArgumentParser(prog="tlseval", epilog=EPILOG,
                                 description="Evaluate TLS individual-tree segmentation.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"tlseval {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score one plot")
    s.add_argument("input", help="LAZ/LAS with treeID and predID")
    s.add_argument("--output", "-o", default="evaluation_results.csv")
    _common(s)

    b = sub.add_parser("batch", help="score many plots")
    b.add_argument("predictions", help="directory of prediction clouds")
    b.add_argument("--reference", "-r", default=None,
                   help="directory of reference clouds; omit if predictions "
                        "already carry the ground-truth field")
    b.add_argument("--out", "-o", default="results", help="output directory")
    b.add_argument("--jobs", "-j", type=int, default=1, help="worker processes")
    b.add_argument("--tolerance", type=float, default=0.05,
                   help="metres, nearest-neighbour label transfer (default: 0.05)")
    _common(b)

    t = sub.add_parser("transfer", help="move predicted labels onto a reference cloud")
    t.add_argument("prediction"); t.add_argument("reference")
    t.add_argument("--output", "-o", required=True)
    t.add_argument("--pred-field", default=DEFAULTS["pred_field"])
    t.add_argument("--tolerance", type=float, default=0.05)
    t.add_argument("--dry-run", action="store_true",
                   help="report coverage without writing")

    r = sub.add_parser("report", help="full analysis of a batch run")
    r.add_argument("results", help="directory written by `tlseval batch`")
    r.add_argument("--attributes", "-a", default=None,
                   help="CSV of per-plot attributes, joined on plot name")
    r.add_argument("--out", "-o", default="report")
    r.add_argument("--strip-suffix", default=None,
                   help="trailing text to remove from plot names before joining "
                        "attributes, e.g. '_mymethod' for Plot_A_mymethod.laz")
    r.add_argument("--no-figures", action="store_true")

    c = sub.add_parser("check", help="compare the config headers of result files")
    c.add_argument("files", nargs="+")
    return ap


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_score(a):
    cfg = config_from(a.voxel_size, a.gt_field, a.pred_field, a.all_trees,
                      a.dominance_threshold, a.fragment_threshold)
    df = evaluate(a.input, a.voxel_size, a.gt_field, a.pred_field, a.all_trees,
                  a.dominance_threshold, a.fragment_threshold)
    if df.empty:
        print("No trees evaluated.", file=sys.stderr)
        return 1
    write_results(df, a.output, cfg)
    s = summarise(df)

    excluded = df.attrs.get("n_excluded_boundary", 0)
    field = df.attrs.get("inside_field")
    print(f"\nvoxel size        {a.voxel_size} m")
    print(f"trees evaluated   {s['n_trees']}"
          + (f"   ({excluded} boundary-clipped excluded via '{field}')" if excluded else ""))
    if field is None and not a.all_trees:
        print("                  (no boundary flag in this file; all trees scored)")
    print(f"pred. instances   {s['n_pred_instances']}")
    print(f"mean IoU          {s['mean_iou']:.3f}")
    print(f"detection rate    {s['detection_rate']:.3f}   [IoU >= {DETECTION_IOU_THRESHOLD}]")
    print(f"matched (any IoU) {s['matched_rate']:.3f}")
    print(f"mean precision    {s['mean_precision']:.3f}")
    print(f"mean recall       {s['mean_recall']:.3f}")
    print("\nfailure events per 100 reference trees:")
    print(f"  missed {s['missed_per100']:.1f}   split {s['split_per100']:.1f}"
          f"   merged {s['merged_per100']:.1f}")
    print(f"\nsaved to {a.output}")
    return 0


def cmd_batch(a):
    from .batch import run
    run(a.predictions, reference=a.reference, out_dir=a.out, jobs=a.jobs,
        tolerance=a.tolerance, voxel_size=a.voxel_size, gt_field=a.gt_field,
        pred_field=a.pred_field, all_trees=a.all_trees,
        dominance_threshold=a.dominance_threshold,
        fragment_threshold=a.fragment_threshold)
    return 0


def cmd_transfer(a):
    from .transfer import transfer_labels, transfer_report
    rep = transfer_report(a.prediction, a.reference, a.pred_field, a.tolerance)
    print(f"reference points   {rep['n_reference_points']:,}")
    print(f"predicted points   {rep['n_predicted_points']:,}")
    print(f"predicted instances{rep['n_predicted_instances']:>8,}")
    print(f"assigned fraction  {rep['assigned_fraction']:.3f}   "
          f"(tolerance {rep['tolerance']} m)")
    if "median_nn_distance" in rep:
        print(f"nn distance        median {rep['median_nn_distance']:.3f} m, "
              f"p95 {rep['p95_nn_distance']:.3f} m")
    if rep["assigned_fraction"] < 0.5:
        print("\nwarning: under half the reference points received a label. Usually the\n"
              "         tolerance is too tight for the prediction's resolution -- compare\n"
              "         it against the p95 nearest-neighbour distance above.", file=sys.stderr)
    if a.dry_run:
        return 0
    out = transfer_labels(a.prediction, a.reference, a.pred_field, a.output, a.tolerance)
    print(f"\nwritten to {out}")
    return 0


def cmd_report(a):
    from .report import build
    build(a.results, attributes=a.attributes, out_dir=a.out,
          figures=not a.no_figures, strip_suffix=a.strip_suffix)
    return 0


def cmd_check(a):
    configs = {}
    for f in a.files:
        try:
            _, cfg = read_results(f)
        except Exception as exc:
            print(f"{f}: unreadable ({exc})", file=sys.stderr)
            return 2
        configs[f] = cfg
        print(f"{f}\n  {json.dumps(cfg, sort_keys=True) if cfg else 'NO CONFIG HEADER'}")
    seen = {json.dumps(c, sort_keys=True) for c in configs.values()}
    if None in configs.values():
        print("\nat least one file carries no config header; it predates config "
              "pinning and cannot be safely compared.", file=sys.stderr)
        return 1
    if len(seen) > 1:
        print("\nMISMATCH: these files were produced under different settings and are "
              "not comparable.", file=sys.stderr)
        keys = set().union(*(c.keys() for c in configs.values()))
        for k in sorted(keys):
            vals = {json.dumps(c.get(k)) for c in configs.values()}
            if len(vals) > 1:
                print(f"  {k}: " + ", ".join(sorted(vals)), file=sys.stderr)
        return 1
    print(f"\nOK: {len(a.files)} file(s) share one configuration.")
    return 0


def main(argv=None):
    a = build_parser().parse_args(argv)
    cmds = {"score": cmd_score, "batch": cmd_batch, "transfer": cmd_transfer,
            "report": cmd_report, "check": cmd_check}
    try:
        return cmds[a.cmd](a)
    except TlsEvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
