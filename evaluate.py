#!/usr/bin/env python3
"""
Tree Instance Segmentation Evaluation

Evaluates predicted tree instance segmentation against ground truth
using voxel-based IoU with Hungarian matching, and classifies each
reference tree's failure mode as Missed / Split / Merged.

Input: a single LAZ/LAS file containing at minimum:
  - x, y, z coordinates
  - treeID:  ground truth instance labels (0 = unlabeled)
  - predID:  predicted instance labels (0 = unlabeled)

Optional field:
  - completely_inside: binary flag (1 = tree fully within plot boundary)
    When present, only trees with completely_inside == 1 are evaluated
    unless --all-trees is set.

Every run writes an `evaluation_config` header into the output CSV.
Results produced under different settings are not comparable, so the
settings travel with the numbers.

Usage:
    python evaluate.py input.laz
    python evaluate.py input.laz --voxel-size 0.05
    python evaluate.py input.laz --all-trees
    python evaluate.py input.laz -o results.csv
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import laspy
from scipy.optimize import linear_sum_assignment


# ── Evaluation configuration ─────────────────────────────────────────────────
#
# Voxel size changes IoU by enough to reorder methods: on TreeScanPL10K a 2 cm
# grid scored 0.01-0.03 higher than a 10 cm grid for the same predictions, which
# is larger than the gap between adjacent methods. Comparing numbers produced at
# different voxel sizes is therefore meaningless, and the failure is silent.
# The config below is stamped into every output file so a mismatch is visible.

# Detection rate threshold. Fixed at 0.5 to match Eq. 3 in the paper:
# a ground-truth tree counts as detected iff its matched prediction
# achieves IoU >= DETECTION_IOU_THRESHOLD.
DETECTION_IOU_THRESHOLD = 0.5

DEFAULTS = {
    "voxel_size": 0.10,      # metres
    "dominance_threshold": 0.5,   # T: share of a tree's points for a prediction to be "dominant"
    "fragment_threshold": 0.1,    # S: share above which a leaked fragment counts as a Split
    "gt_field": "treeID",
    "pred_field": "predID",
    "all_trees": False,
}


# ── Voxelization ─────────────────────────────────────────────────────────────

def encode_voxels(coords: np.ndarray, voxel_size: float) -> np.ndarray:
    """Quantize points to voxels and encode as unique 64-bit integers."""
    q = np.floor(coords / voxel_size).astype(np.int64)
    q -= q.min(axis=0)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


# ── Instance building ────────────────────────────────────────────────────────

def build_instances(voxel_codes: np.ndarray, instance_ids: np.ndarray) -> dict:
    """
    Group voxel codes by instance ID.

    Returns dict: instance_id -> sorted array of unique voxel codes.
    IDs <= 0 are ignored (unlabeled points).
    """
    mask = instance_ids > 0
    ids = instance_ids[mask]
    codes = voxel_codes[mask]

    order = np.argsort(ids, kind="mergesort")
    ids, codes = ids[order], codes[order]
    unique_ids, splits = np.unique(ids, return_index=True)

    return {
        int(uid): np.unique(chunk)
        for uid, chunk in zip(unique_ids, np.split(codes, splits[1:]))
    }


# ── Matching ─────────────────────────────────────────────────────────────────

def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two sorted arrays of voxel codes."""
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter) if inter > 0 else 0.0


def match_instances(gt: dict, pred: dict) -> dict:
    """Hungarian matching maximizing IoU. Returns {gt_id: pred_id}."""
    gt_ids = list(gt)
    pred_ids = list(pred)
    if not gt_ids or not pred_ids:
        return {}

    cost = np.ones((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for i, g in enumerate(gt_ids):
        for j, p in enumerate(pred_ids):
            iou = compute_iou(gt[g], pred[p])
            if iou > 0:
                cost[i, j] = 1.0 - iou

    ri, ci = linear_sum_assignment(cost)
    return {
        gt_ids[r]: pred_ids[c]
        for r, c in zip(ri, ci)
        if cost[r, c] < 1.0
    }


# ── Failure taxonomy ─────────────────────────────────────────────────────────

def classify_failures(
    gt_ids: np.ndarray,
    pred_ids: np.ndarray,
    tree_ids: list,
    heights: dict,
    dominance_threshold: float,
    fragment_threshold: float,
) -> dict:
    """
    Classify each reference tree as Missed / Split / Merged.

    Let c(i, j) be the share of tree i's points that fall in prediction j,
    with j = 0 denoting background. A prediction is *dominant* for tree i
    when c(i, j) > T. With thresholds T and S:

      Missed  c(i, 0) > T
              Most of the tree was left as background.

      Split   some prediction j with c(i, j) > S is not dominant for any tree.
              A fragment of the tree leaked into a prediction that represents
              no real tree.

      Merged  the tree's dominant prediction is also dominant for a taller
              reference tree, i.e. it was absorbed into a larger neighbour.

    Flags are independent; a tree may carry more than one.

    These shares are computed on points rather than voxels, so unlike IoU the
    taxonomy does not shift with voxel size.

    Returns {tree_id: {"missed": bool, "split": bool, "merged": bool}}.
    """
    shares, dominant = {}, {}
    for tid in tree_ids:
        sel = gt_ids == tid
        total = int(sel.sum())
        if total == 0:
            shares[tid] = {}
            continue
        labels, counts = np.unique(pred_ids[sel], return_counts=True)
        shares[tid] = {int(a): b / total for a, b in zip(labels, counts)}

        foreground = [(v, k) for k, v in shares[tid].items() if k > 0]
        if foreground:
            best_share, best_id = max(foreground)
            if best_share > dominance_threshold:
                dominant[tid] = best_id

    dominant_preds = set(dominant.values())

    out = {}
    for tid in tree_ids:
        s = shares.get(tid, {})
        missed = s.get(0, 0.0) > dominance_threshold
        split = any(
            pid > 0 and share > fragment_threshold and pid not in dominant_preds
            for pid, share in s.items()
        )

        merged = False
        own = dominant.get(tid)
        if own is not None:
            for other, other_pred in dominant.items():
                if (other != tid and other_pred == own
                        and heights.get(other, 0.0) > heights.get(tid, 0.0)):
                    merged = True
                    break

        out[tid] = {"missed": missed, "split": split, "merged": merged}
    return out


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    path: str,
    voxel_size: float = DEFAULTS["voxel_size"],
    gt_field: str = DEFAULTS["gt_field"],
    pred_field: str = DEFAULTS["pred_field"],
    all_trees: bool = DEFAULTS["all_trees"],
    dominance_threshold: float = DEFAULTS["dominance_threshold"],
    fragment_threshold: float = DEFAULTS["fragment_threshold"],
) -> pd.DataFrame:
    """
    Run full evaluation on a single point cloud file.

    Returns a DataFrame with one row per evaluated ground-truth tree:
        treeID, matched_predID, iou, precision, recall,
        gt_voxel_count, pred_voxel_count, missed, split, merged
    """
    # ── Load ──
    las = laspy.read(path)
    xyz = np.stack([las.x, las.y, las.z], axis=1)

    for field in (gt_field, pred_field):
        try:
            las[field]
        except Exception:
            raise SystemExit(
                f"error: field '{field}' not found in {path}\n"
                f"       available: {', '.join(d.name for d in las.point_format.dimensions)}"
            )

    gt_ids = np.asarray(las[gt_field]).astype(np.int64)
    pred_ids = np.asarray(las[pred_field]).astype(np.int64)

    try:
        completely_inside = np.asarray(las["completely_inside"])
    except Exception:
        completely_inside = None

    # ── Voxelize & build instances ──
    voxel_codes = encode_voxels(xyz, voxel_size)
    gt_inst = build_instances(voxel_codes, gt_ids)
    pred_inst = build_instances(voxel_codes, pred_ids)

    # ── Select trees to evaluate ──
    tree_ids = sorted(gt_inst)
    if not all_trees and completely_inside is not None:
        inside_trees = set()
        for tid in tree_ids:
            if np.any(completely_inside[gt_ids == tid] == 1):
                inside_trees.add(tid)
        tree_ids = sorted(inside_trees)

    # ── Match ──
    matches = match_instances(gt_inst, pred_inst)

    # ── Failure taxonomy ──
    # Heights are measured from the cloud, never joined from a side table:
    # a tree's attributes must come from the same labels its score does.
    heights = {}
    for tid in gt_inst:
        z = xyz[gt_ids == tid][:, 2]
        heights[tid] = float(z.max() - z.min()) if z.size else 0.0

    failures = classify_failures(
        gt_ids, pred_ids, sorted(gt_inst), heights,
        dominance_threshold, fragment_threshold,
    )

    # ── Per-tree metrics ──
    rows = []
    for tid in tree_ids:
        gt_vox = gt_inst[tid]
        pid = matches.get(tid)
        flags = failures.get(tid, {"missed": False, "split": False, "merged": False})

        if pid is None:
            rows.append(dict(treeID=tid, matched_predID=-1,
                             iou=0.0, precision=0.0, recall=0.0,
                             gt_voxel_count=gt_vox.size, pred_voxel_count=0,
                             missed=int(flags["missed"]), split=int(flags["split"]),
                             merged=int(flags["merged"])))
            continue

        pred_vox = pred_inst[pid]
        inter = np.intersect1d(gt_vox, pred_vox, assume_unique=True).size
        union = gt_vox.size + pred_vox.size - inter

        rows.append(dict(
            treeID=tid,
            matched_predID=pid,
            iou=inter / union if union > 0 else 0.0,
            precision=inter / pred_vox.size if pred_vox.size > 0 else 0.0,
            recall=inter / gt_vox.size if gt_vox.size > 0 else 0.0,
            gt_voxel_count=gt_vox.size,
            pred_voxel_count=pred_vox.size,
            missed=int(flags["missed"]),
            split=int(flags["split"]),
            merged=int(flags["merged"]),
        ))

    df = pd.DataFrame(rows)
    df.attrs["n_pred_instances"] = len(pred_inst)
    return df


# ── Output ───────────────────────────────────────────────────────────────────

def write_results(df: pd.DataFrame, path: str, config: dict) -> None:
    """Write results with the evaluation config as a leading comment line."""
    with open(path, "w", newline="") as fh:
        fh.write("# evaluation_config " + json.dumps(config, sort_keys=True) + "\n")
        df.to_csv(fh, index=False)


def read_results(path: str):
    """Read a results CSV written by this script. Returns (DataFrame, config)."""
    with open(path) as fh:
        first = fh.readline()
    config = None
    if first.startswith("# evaluation_config "):
        config = json.loads(first[len("# evaluation_config "):])
        return pd.read_csv(path, skiprows=1), config
    return pd.read_csv(path), config


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tree instance segmentation (voxel-based IoU).",
        epilog="Results are only comparable across runs that share a voxel size; "
               "the settings used are recorded in the output file.",
    )
    parser.add_argument("input", help="LAZ/LAS file with treeID and predID fields")
    parser.add_argument("--voxel-size", "-v", type=float, default=DEFAULTS["voxel_size"],
                        help=f"Voxel size in meters (default: {DEFAULTS['voxel_size']})")
    parser.add_argument("--gt-field", default=DEFAULTS["gt_field"],
                        help=f"Ground truth instance field (default: {DEFAULTS['gt_field']})")
    parser.add_argument("--pred-field", default=DEFAULTS["pred_field"],
                        help=f"Predicted instance field (default: {DEFAULTS['pred_field']})")
    parser.add_argument("--all-trees", action="store_true",
                        help="Evaluate all trees (ignore completely_inside filter)")
    parser.add_argument("--dominance-threshold", type=float,
                        default=DEFAULTS["dominance_threshold"],
                        help="T: share of a tree's points making a prediction dominant "
                             f"(default: {DEFAULTS['dominance_threshold']})")
    parser.add_argument("--fragment-threshold", type=float,
                        default=DEFAULTS["fragment_threshold"],
                        help="S: share above which a leaked fragment counts as a Split "
                             f"(default: {DEFAULTS['fragment_threshold']})")
    parser.add_argument("--output", "-o", default="evaluation_results.csv",
                        help="Output CSV path (default: evaluation_results.csv)")
    args = parser.parse_args()

    config = {
        "voxel_size": args.voxel_size,
        "dominance_threshold": args.dominance_threshold,
        "fragment_threshold": args.fragment_threshold,
        "gt_field": args.gt_field,
        "pred_field": args.pred_field,
        "all_trees": args.all_trees,
    }

    df = evaluate(
        args.input, args.voxel_size, args.gt_field, args.pred_field,
        args.all_trees, args.dominance_threshold, args.fragment_threshold,
    )
    write_results(df, args.output, config)

    # ── Summary ──
    n = len(df)
    if n == 0:
        print("No trees evaluated.", file=sys.stderr)
        return

    matched = (df["matched_predID"] != -1).sum()
    # A detection is a match at IoU >= 0.5, following the convention used in the
    # literature. Counting merely-matched trees instead inflates this number:
    # Hungarian matching pairs a tree with its best prediction however poor.
    detected = (df["iou"] >= DETECTION_IOU_THRESHOLD).sum()
    print(f"\nVoxel size:       {args.voxel_size} m")
    print(f"Trees evaluated:  {n}")
    print(f"Pred. instances:  {df.attrs.get('n_pred_instances', 'n/a')}")
    print(f"Detection rate:   {detected}/{n} ({detected/n:.3f})   [IoU >= {DETECTION_IOU_THRESHOLD}]")
    print(f"Matched (any IoU):{matched}/{n} ({matched/n:.3f})")
    print(f"Mean IoU:         {df['iou'].mean():.3f}")
    print(f"Mean Precision:   {df['precision'].mean():.3f}")
    print(f"Mean Recall:      {df['recall'].mean():.3f}")
    print("\nFailure events (per 100 reference trees):")
    print(f"  Missed:         {100 * df['missed'].mean():.1f}")
    print(f"  Split:          {100 * df['split'].mean():.1f}")
    print(f"  Merged:         {100 * df['merged'].mean():.1f}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
