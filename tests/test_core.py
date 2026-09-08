"""Fixtures whose correct answers can be worked out by hand.

Every expected value below is derived in the comment above it. A test that only
compares against whatever the code currently returns cannot catch a regression
in the thing it is testing, which is how the two defects this package was built
to fix survived as long as they did.
"""

import numpy as np
import pytest

from tlseval.core import (build_instances, classify_failures, compute_iou,
                          encode_voxels, evaluate, match_instances)

pytest.importorskip("laspy")
import laspy  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def write_cloud(path, xyz, tree_ids, pred_ids, inside=None, inside_name="completelyInside"):
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.offsets, hdr.scales = np.min(xyz, axis=0), np.array([0.001] * 3)
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for name, vals in (("treeID", tree_ids), ("predID", pred_ids)):
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.int32))
        las[name] = np.asarray(vals, dtype=np.int32)
    if inside is not None:
        las.add_extra_dim(laspy.ExtraBytesParams(name=inside_name, type=np.int32))
        las[inside_name] = np.asarray(inside, dtype=np.int32)
    las.write(str(path))
    return str(path)


def column(x, y, n=40, z0=0.0, dz=0.25):
    """A vertical stick of n points at (x, y)."""
    return np.column_stack([np.full(n, x), np.full(n, y), z0 + dz * np.arange(n)])


# ── voxelisation and set maths ───────────────────────────────────────────────

def test_encode_voxels_is_injective_per_cell():
    # Three points: two inside one 1 m cell, one in the next cell along x.
    xyz = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [1.5, 0.1, 0.1]])
    codes = encode_voxels(xyz, 1.0)
    assert codes[0] == codes[1] != codes[2]


def test_compute_iou_by_hand():
    # |A| = 4, |B| = 4, |A n B| = 2  ->  IoU = 2 / (4 + 4 - 2) = 1/3
    a = np.array([1, 2, 3, 4]); b = np.array([3, 4, 5, 6])
    assert compute_iou(a, b) == pytest.approx(1 / 3)
    assert compute_iou(a, np.array([9, 10])) == 0.0
    assert compute_iou(a, a) == 1.0


def test_build_instances_ignores_unlabelled():
    codes = np.array([10, 11, 12, 13])
    inst = build_instances(codes, np.array([0, 1, 1, 2]))
    assert set(inst) == {1, 2}                    # label 0 dropped
    assert inst[1].tolist() == [11, 12]
    assert inst[2].tolist() == [13]


def test_build_instances_deduplicates_voxels():
    # Two points in the same voxel contribute one voxel, not two.
    inst = build_instances(np.array([7, 7, 8]), np.array([1, 1, 1]))
    assert inst[1].tolist() == [7, 8]


# ── matching ─────────────────────────────────────────────────────────────────

def test_matching_is_one_to_one_and_maximises_iou():
    # gt1 overlaps predA fully; gt2 overlaps predB fully. The greedy choice and
    # the optimal choice agree here, so only the pairing is under test.
    gt = {1: np.array([1, 2, 3]), 2: np.array([7, 8, 9])}
    pred = {10: np.array([1, 2, 3]), 20: np.array([7, 8, 9])}
    assert match_instances(gt, pred) == {1: 10, 2: 20}


def test_matching_prefers_globally_best_assignment():
    # predA overlaps gt1 by 2 voxels and gt2 by 3. Greedy would give predA to
    # gt2, leaving gt1 unmatched. The optimal assignment pairs
    # gt1-predA (IoU 2/4) and gt2-predB (IoU 3/3), total 1.5, which beats
    # gt2-predA (3/5) with gt1 unmatched.
    gt = {1: np.array([1, 2, 3]), 2: np.array([4, 5, 6])}
    pred = {10: np.array([1, 2, 4, 5, 6]), 20: np.array([4, 5, 6])}
    m = match_instances(gt, pred)
    assert len(set(m.values())) == len(m)          # one-to-one
    assert m[2] == 20

def test_matching_returns_nothing_when_a_side_is_empty():
    assert match_instances({}, {1: np.array([1])}) == {}
    assert match_instances({1: np.array([1])}, {}) == {}


# ── failure taxonomy ─────────────────────────────────────────────────────────

def test_missed_when_most_points_are_background():
    # Tree 1 has 10 points, 9 of them predicted 0 -> c(1,0) = 0.9 > T = 0.5
    gt = np.array([1] * 10)
    pred = np.array([0] * 9 + [5])
    f = classify_failures(gt, pred, [1], {1: 10.0}, 0.5, 0.1)
    assert f[1]["missed"] is True


def test_split_when_a_fragment_lands_in_a_non_tree_prediction():
    # Tree 1: 10 points, 7 in pred 5 (dominant, 0.7 > 0.5), 3 in pred 9.
    # pred 9 holds 0.3 > S = 0.1 of the tree and is dominant for no tree -> Split.
    gt = np.array([1] * 10)
    pred = np.array([5] * 7 + [9] * 3)
    f = classify_failures(gt, pred, [1], {1: 10.0}, 0.5, 0.1)
    assert f[1]["split"] is True
    assert f[1]["missed"] is False


def test_no_split_when_the_fragment_is_below_threshold():
    # Only 1 of 20 points leaks: 0.05 < S = 0.1, so this is not a Split.
    gt = np.array([1] * 20)
    pred = np.array([5] * 19 + [9])
    assert classify_failures(gt, pred, [1], {1: 10.0}, 0.5, 0.1)[1]["split"] is False


def test_merged_only_flags_the_shorter_tree():
    # Both trees are dominated by prediction 5. Tree 2 is taller (20 m vs 10 m),
    # so tree 1 was absorbed into tree 2, not the other way round.
    gt = np.array([1] * 10 + [2] * 10)
    pred = np.array([5] * 20)
    f = classify_failures(gt, pred, [1, 2], {1: 10.0, 2: 20.0}, 0.5, 0.1)
    assert f[1]["merged"] is True
    assert f[2]["merged"] is False


def test_taxonomy_is_independent_of_voxel_size(tmp_path):
    # Shares are computed on points, so the flags must be identical at any grid.
    xyz = np.vstack([column(0, 0), column(3, 0)])
    tree = np.array([1] * 40 + [2] * 40)
    pred = np.array([5] * 80)                       # both trees fused into one
    p = write_cloud(tmp_path / "t.laz", xyz, tree, pred)
    flags = [tuple(evaluate(p, voxel_size=v)[["missed", "split", "merged"]].sum())
             for v in (0.05, 0.5, 2.0)]
    assert len(set(flags)) == 1


# ── end to end ───────────────────────────────────────────────────────────────

def test_perfect_segmentation_scores_one(tmp_path):
    xyz = np.vstack([column(0, 0), column(5, 0)])
    ids = np.array([1] * 40 + [2] * 40)
    df = evaluate(write_cloud(tmp_path / "p.laz", xyz, ids, ids), voxel_size=0.5)
    assert len(df) == 2
    assert df["iou"].tolist() == [1.0, 1.0]
    assert df[["missed", "split", "merged"]].to_numpy().sum() == 0


def test_unmatched_tree_scores_zero_not_dropped(tmp_path):
    # Tree 2 is predicted entirely as background. It must appear in the output
    # with IoU 0 -- dropping it inflates the mean, which is precisely the
    # convention error this package exists to avoid.
    xyz = np.vstack([column(0, 0), column(5, 0)])
    tree = np.array([1] * 40 + [2] * 40)
    pred = np.array([1] * 40 + [0] * 40)
    df = evaluate(write_cloud(tmp_path / "u.laz", xyz, tree, pred), voxel_size=0.5)
    assert len(df) == 2, "unmatched trees must stay in the table"
    row = df[df.treeID == 2].iloc[0]
    assert row["iou"] == 0.0 and row["matched_predID"] == -1
    assert row["missed"] == 1
    assert df["iou"].mean() == pytest.approx(0.5)   # (1.0 + 0.0) / 2


@pytest.mark.parametrize("field", ["completely_inside", "completelyInside"])
def test_boundary_filter_accepts_both_spellings(tmp_path, field):
    # The reference dataset ships `completelyInside`. Looking only for
    # `completely_inside` made the filter a silent no-op and scored
    # boundary-clipped trees as if they were whole.
    xyz = np.vstack([column(0, 0), column(5, 0)])
    ids = np.array([1] * 40 + [2] * 40)
    inside = np.array([1] * 40 + [0] * 40)          # tree 2 is on the boundary
    p = write_cloud(tmp_path / f"{field}.laz", xyz, ids, ids, inside, field)
    assert evaluate(p, voxel_size=0.5)["treeID"].tolist() == [1]
    assert evaluate(p, voxel_size=0.5, all_trees=True)["treeID"].tolist() == [1, 2]


def test_extra_predictions_cannot_raise_the_score(tmp_path):
    # predID is a per-point label, so instances partition the points: an extra
    # instance can only be carved out of an existing one. Splitting a perfect
    # prediction must therefore lower its IoU, never raise it.
    xyz = column(0, 0, n=40)
    tree = np.ones(40, dtype=int)
    whole = evaluate(write_cloud(tmp_path / "w.laz", xyz, tree, np.ones(40, dtype=int)),
                     voxel_size=0.5)["iou"].iloc[0]
    carved = np.where(np.arange(40) < 30, 1, 2)
    split = evaluate(write_cloud(tmp_path / "s.laz", xyz, tree, carved),
                     voxel_size=0.5)["iou"].iloc[0]
    assert whole == pytest.approx(1.0)
    assert split < whole


def test_config_header_round_trips(tmp_path):
    from tlseval.core import config_from, read_results, write_results
    import pandas as pd
    cfg = config_from(voxel_size=0.07)
    p = tmp_path / "r.csv"
    write_results(pd.DataFrame({"treeID": [1], "iou": [0.5]}), p, cfg)
    df, back = read_results(p)
    assert back == cfg and len(df) == 1


# ── transfer and batch ────────────────────────────────────────────────────

def test_transfer_moves_labels_onto_reference_points(tmp_path):
    from tlseval.transfer import transfer_labels, transfer_report
    ref_xyz = np.vstack([column(0, 0, n=20), column(4, 0, n=20)])
    ids = np.array([1] * 20 + [2] * 20)
    ref = write_cloud(tmp_path / "ref.laz", ref_xyz, ids, np.zeros(40, int))

    # Prediction on shifted, halved points: labels must still land correctly.
    pred_xyz = ref_xyz[::2] + 0.01
    pred = write_cloud(tmp_path / "pred.laz", pred_xyz,
                       np.zeros(20, int), ids[::2])

    rep = transfer_report(pred, ref, tolerance=0.5)
    assert rep["assigned_fraction"] == 1.0
    out = transfer_labels(pred, ref, out_path=tmp_path / "out.laz", tolerance=0.5)
    got = laspy.read(out)
    assert np.array_equal(np.asarray(got["predID"]), ids)


def test_transfer_tolerance_rejects_distant_points(tmp_path):
    from tlseval.transfer import transfer_report
    ref_xyz = column(0, 0, n=20)
    ref = write_cloud(tmp_path / "r2.laz", ref_xyz, np.ones(20, int), np.zeros(20, int))
    far = write_cloud(tmp_path / "p2.laz", ref_xyz + 5.0,
                      np.zeros(20, int), np.ones(20, int))
    assert transfer_report(far, ref, tolerance=0.1)["assigned_fraction"] == 0.0


def test_batch_survives_a_broken_plot(tmp_path):
    from tlseval.batch import run
    d = tmp_path / "preds"; d.mkdir()
    xyz = np.vstack([column(0, 0), column(5, 0)])
    ids = np.array([1] * 40 + [2] * 40)
    write_cloud(d / "good.laz", xyz, ids, ids)
    # No predID field: this plot must fail without taking the run down.
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.offsets, hdr.scales = xyz.min(axis=0), np.array([0.001] * 3)
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    las["treeID"] = ids.astype(np.int32)
    las.write(str(d / "broken.laz"))

    per_tree, per_plot, overall = run(d, out_dir=tmp_path / "out",
                                      voxel_size=0.5, progress=False)
    assert len(per_plot) == 1 and per_plot.iloc[0]["plot"] == "good"
    assert overall.iloc[0]["n_failed_plots"] == 1
    assert (tmp_path / "out" / "failures.csv").exists()


def test_batch_output_carries_the_config_header(tmp_path):
    from tlseval.batch import run
    from tlseval.core import read_results
    d = tmp_path / "p"; d.mkdir()
    xyz = column(0, 0)
    write_cloud(d / "a.laz", xyz, np.ones(40, int), np.ones(40, int))
    run(d, out_dir=tmp_path / "o", voxel_size=0.33, progress=False)
    for name in ("per_tree", "per_plot", "summary"):
        _, cfg = read_results(tmp_path / "o" / f"{name}.csv")
        assert cfg["voxel_size"] == 0.33
