"""A synthetic plot and a naive segmenter, so the tools can be run in seconds.

Nothing here is a research contribution. It exists so that `tlseval demo`
works on a fresh clone with no download, and so the tutorial can show a real
`predID` being produced rather than handing the reader one.

The baseline is deliberately simple and deliberately imperfect: it finds stems
by clustering points in a slice above the ground and grows each crown by
nearest stem. It misses suppressed trees and fuses touching crowns, which makes
the failure taxonomy show something instead of a row of zeros.
"""

import numpy as np

RNG_SEED = 20260908


def make_plot(n_trees=18, radius=15.0, seed=RNG_SEED, point_spacing=0.06,
              understory_fraction=0.3):
    """Generate a circular plot of cone-crowned trees over a ground plane.

    Returns (xyz, treeID, completelyInside). Trees whose crown crosses the plot
    edge are flagged 0, mirroring how the reference dataset is built: those
    trees are clipped, so scoring them would penalise a method for points that
    were never in the cloud.

    A share of the stand is understory -- short trees growing close to a
    dominant neighbour. They are what makes the plot non-trivial: a stem-slice
    segmenter tends to miss them or fuse them into the tree above, so the
    Missed and Merged columns show something rather than a row of zeros.
    """
    rng = np.random.default_rng(seed)
    xs, ys, zs, ids = [], [], [], []

    # Ground: a thin noisy sheet, labelled 0. Every real cloud has one and a
    # segmenter has to cope with it.
    n_ground = int(np.pi * radius ** 2 / (point_spacing * 8) ** 2)
    ga = rng.uniform(0, 2 * np.pi, n_ground)
    gr = radius * np.sqrt(rng.uniform(0, 1, n_ground))
    xs.append(gr * np.cos(ga)); ys.append(gr * np.sin(ga))
    zs.append(rng.normal(0, 0.02, n_ground)); ids.append(np.zeros(n_ground, int))

    n_under = int(round(n_trees * understory_fraction))
    dominant_centres = []
    centres, inside_flags = [], []
    for tid in range(1, n_trees + 1):
        understory = tid > n_trees - n_under and dominant_centres

        if understory:
            # Tuck it close under an existing dominant crown.
            px, py = dominant_centres[rng.integers(len(dominant_centres))]
            off = rng.uniform(0.6, 1.8)
            ang0 = rng.uniform(0, 2 * np.pi)
            cx, cy = px + off * np.cos(ang0), py + off * np.sin(ang0)
            height = rng.uniform(3.5, 7.5)
        else:
            for _ in range(200):
                a, rr = rng.uniform(0, 2 * np.pi), radius * np.sqrt(rng.uniform(0, 1))
                cx, cy = rr * np.cos(a), rr * np.sin(a)
                if all((cx - px) ** 2 + (cy - py) ** 2 > 2.6 ** 2 for px, py in centres):
                    break
            height = rng.uniform(11, 26)
            dominant_centres.append((cx, cy))
        centres.append((cx, cy))

        crown_r = np.clip(height * rng.uniform(0.16, 0.28), 0.8, 4.4)
        crown_base = height * rng.uniform(0.28, 0.45)

        # Stem
        n_stem = max(40, int((crown_base) / point_spacing))
        sz = np.linspace(0, crown_base, n_stem)
        jitter = rng.normal(0, 0.03, (n_stem, 2))
        xs.append(cx + jitter[:, 0]); ys.append(cy + jitter[:, 1]); zs.append(sz)
        ids.append(np.full(n_stem, tid))

        # Crown: a cone shell, denser near the top
        n_crown = int(2200 * (crown_r / 2.5) ** 2)
        u = rng.uniform(0, 1, n_crown) ** 0.7
        cz = crown_base + u * (height - crown_base)
        rad = crown_r * (1 - u) * rng.uniform(0.55, 1.0, n_crown)
        ang = rng.uniform(0, 2 * np.pi, n_crown)
        xs.append(cx + rad * np.cos(ang)); ys.append(cy + rad * np.sin(ang))
        zs.append(cz + rng.normal(0, 0.06, n_crown)); ids.append(np.full(n_crown, tid))

        inside_flags.append(np.hypot(cx, cy) + crown_r <= radius)

    xyz = np.column_stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)])
    tree = np.concatenate(ids)
    inside = np.zeros(len(tree), dtype=np.int32)
    for tid, flag in enumerate(inside_flags, start=1):
        if flag:
            inside[tree == tid] = 1

    # Clip to the plot circle so edge trees really are cut, not merely flagged.
    keep = np.hypot(xyz[:, 0], xyz[:, 1]) <= radius
    return xyz[keep], tree[keep], inside[keep]


def naive_segment(xyz, ground_cut=1.3, slice_hi=3.0, stem_eps=0.6, min_stem=12):
    """A minimal baseline: cluster a stem slice, then assign crowns by nearest stem.

    This is the shape almost every algorithmic pipeline takes, reduced to its
    essentials. Returns predID (0 = unassigned), so its output can go straight
    into the scorer.
    """
    from scipy.spatial import cKDTree

    pred = np.zeros(len(xyz), dtype=np.int64)
    above = xyz[:, 2] > ground_cut
    if not above.any():
        return pred

    # 1. stem slice
    sl = above & (xyz[:, 2] <= slice_hi)
    if sl.sum() < min_stem:
        return pred
    pts = xyz[sl][:, :2]

    # 2. single-link clustering of the slice, via a radius graph
    tree = cKDTree(pts)
    pairs = tree.query_pairs(stem_eps, output_type="ndarray")
    parent = np.arange(len(pts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(i) for i in range(len(pts))])
    labels, counts = np.unique(roots, return_counts=True)
    keep = labels[counts >= min_stem]
    if not len(keep):
        return pred

    # 3. one seed per cluster, at its centroid
    seeds = np.array([pts[roots == lab].mean(axis=0) for lab in keep])

    # 4. every above-ground point joins its nearest seed
    _, nearest = cKDTree(seeds).query(xyz[above][:, :2], workers=-1)
    pred[above] = nearest + 1
    return pred


def build(out_dir="demo", n_trees=24, seed=RNG_SEED):
    """Write demo/plot.laz carrying treeID, completelyInside and predID."""
    from pathlib import Path

    import laspy

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xyz, tree, inside = make_plot(n_trees=n_trees, seed=seed)
    pred = naive_segment(xyz)

    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.offsets, hdr.scales = xyz.min(axis=0), np.array([0.001] * 3)
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for name, vals in (("treeID", tree), ("completelyInside", inside), ("predID", pred)):
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.int32))
        las[name] = np.asarray(vals, dtype=np.int32)
    path = out / "plot.laz"
    las.write(str(path))
    return str(path), {
        "points": len(xyz),
        "reference_trees": int(tree.max()),
        "completely_inside": int(np.unique(tree[inside == 1]).size),
        "predicted_instances": int(np.unique(pred[pred > 0]).size),
    }
