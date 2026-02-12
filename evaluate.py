#!/usr/bin/env python3
"""
Tree Instance Segmentation Evaluation

Evaluates predicted tree instance segmentation against ground truth
using voxel-based IoU with Hungarian matching.

Input: a single LAZ/LAS file containing at minimum:
  - x, y, z coordinates
  - treeID:  ground truth instance labels (0 = unlabeled)
  - predID:  predicted instance labels (0 = unlabeled)

Optional field:
  - completely_inside: binary flag (1 = tree fully within plot boundary)
    When present, only trees with completely_inside == 1 are evaluated
    unless --all-trees is set.

Usage:
    python evaluate.py input.laz
    python evaluate.py input.laz --voxel-size 0.05
    python evaluate.py input.laz --all-trees
    python evaluate.py input.laz -o results.csv
"""

import argparse
import numpy as np
import pandas as pd
import laspy
from scipy.optimize import linear_sum_assignment


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


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    path: str,
    voxel_size: float = 0.1,
    gt_field: str = "treeID",
    pred_field: str = "predID",
    all_trees: bool = False,
) -> pd.DataFrame:
    """
    Run full evaluation on a single point cloud file.

    Returns a DataFrame with one row per evaluated ground-truth tree:
        treeID, matched_predID, iou, precision, recall,
        gt_voxel_count, pred_voxel_count
    """
    # ── Load ──
    las = laspy.read(path)
    xyz = np.stack([las.x, las.y, las.z], axis=1)
    gt_ids = np.asarray(las[gt_field])
    pred_ids = np.asarray(las[pred_field])

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

    # ── Per-tree metrics ──
    rows = []
    for tid in tree_ids:
        gt_vox = gt_inst[tid]
        pid = matches.get(tid)

        if pid is None:
            rows.append(dict(treeID=tid, matched_predID=-1,
                             iou=0.0, precision=0.0, recall=0.0,
                             gt_voxel_count=gt_vox.size, pred_voxel_count=0))
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
        ))

    return pd.DataFrame(rows)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tree instance segmentation (voxel-based IoU)."
    )
    parser.add_argument("input", help="LAZ/LAS file with treeID and predID fields")
    parser.add_argument("--voxel-size", "-v", type=float, default=0.1,
                        help="Voxel size in meters (default: 0.1)")
    parser.add_argument("--gt-field", default="treeID",
                        help="Ground truth instance field (default: treeID)")
    parser.add_argument("--pred-field", default="predID",
                        help="Prediction instance field (default: predID)")
    parser.add_argument("--all-trees", action="store_true",
                        help="Evaluate all trees (ignore completely_inside filter)")
    parser.add_argument("--output", "-o", default="evaluation_results.csv",
                        help="Output CSV path (default: evaluation_results.csv)")
    args = parser.parse_args()

    df = evaluate(args.input, args.voxel_size, args.gt_field, args.pred_field, args.all_trees)
    df.to_csv(args.output, index=False)

    # ── Summary ──
    n = len(df)
    matched = (df["matched_predID"] != -1).sum()
    print(f"\nTrees evaluated:  {n}")
    print(f"Detection rate:   {matched}/{n} ({matched/n:.3f})" if n > 0 else "")
    print(f"Mean IoU:         {df['iou'].mean():.3f}")
    print(f"Mean Precision:   {df['precision'].mean():.3f}")
    print(f"Mean Recall:      {df['recall'].mean():.3f}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
