"""Move predicted labels onto the reference points.

The scorer builds voxel sets from a single cloud, so ground truth and
prediction must sit on *the same points*. Most segmentation pipelines do not
return them that way: they downsample, filter ground, drop noise, or reorder.
Scoring such output directly compares voxel sets built from different points,
and every metric is affected without anything looking wrong.

This module does the transfer once, correctly, so that each method does not
reimplement it. It is the same nearest-neighbour step used to score every
method in the paper.
"""

import tempfile

import laspy
import numpy as np
from scipy.spatial import cKDTree

from .core import TlsEvalError

DEFAULT_TOLERANCE = 0.05  # metres


def transfer_labels(
    prediction_path,
    reference_path,
    pred_field="predID",
    out_path=None,
    tolerance=DEFAULT_TOLERANCE,
):
    """Write a copy of the reference cloud carrying the prediction's labels.

    Each reference point takes the label of the nearest predicted point, if one
    lies within `tolerance` metres; otherwise it is left unlabelled (0). The
    tolerance matters: too large and ground points inherit a tree's label, too
    small and a legitimately downsampled prediction loses most of its coverage.
    The default suits clouds voxelised at 2 cm.

    Returns the path written. With out_path=None a temporary file is created
    and the caller owns it.
    """
    pred = laspy.read(str(prediction_path))
    ref = laspy.read(str(reference_path))

    try:
        labels = np.asarray(pred[pred_field]).astype(np.int64)
    except Exception:
        raise TlsEvalError(
            f"field '{pred_field}' not found in {prediction_path}; available: "
            + ", ".join(d.name for d in pred.point_format.dimensions)
        )

    pred_xyz = np.stack([pred.x, pred.y, pred.z], axis=1)
    ref_xyz = np.stack([ref.x, ref.y, ref.z], axis=1)

    # Only labelled predicted points can donate a label; unlabelled ones would
    # otherwise win the nearest-neighbour query and erase a real assignment.
    keep = labels > 0
    out = np.zeros(len(ref_xyz), dtype=np.int64)
    if keep.any():
        dist, idx = cKDTree(pred_xyz[keep]).query(ref_xyz, workers=-1)
        within = dist <= tolerance
        out[within] = labels[keep][idx[within]]

    if pred_field in [d.name for d in ref.point_format.dimensions]:
        ref[pred_field] = out
    else:
        ref.add_extra_dim(laspy.ExtraBytesParams(name=pred_field, type=np.int32))
        ref[pred_field] = out.astype(np.int32)

    if out_path is None:
        fh = tempfile.NamedTemporaryFile(suffix=".laz", delete=False)
        out_path, _ = fh.name, fh.close()
    ref.write(str(out_path))
    return str(out_path)


def transfer_report(prediction_path, reference_path, pred_field="predID",
                    tolerance=DEFAULT_TOLERANCE) -> dict:
    """Diagnose a transfer without writing anything.

    Use this when a method scores far worse than expected: a low
    `assigned_fraction` usually means the tolerance is wrong for the
    prediction's resolution, not that the segmentation is bad.
    """
    pred = laspy.read(str(prediction_path))
    ref = laspy.read(str(reference_path))
    labels = np.asarray(pred[pred_field]).astype(np.int64)
    pred_xyz = np.stack([pred.x, pred.y, pred.z], axis=1)
    ref_xyz = np.stack([ref.x, ref.y, ref.z], axis=1)

    keep = labels > 0
    if not keep.any():
        return {"n_reference_points": len(ref_xyz), "n_predicted_points": len(pred_xyz),
                "n_predicted_instances": 0, "assigned_fraction": 0.0}
    dist, _ = cKDTree(pred_xyz[keep]).query(ref_xyz, workers=-1)
    return {
        "n_reference_points": len(ref_xyz),
        "n_predicted_points": len(pred_xyz),
        "n_predicted_instances": int(np.unique(labels[keep]).size),
        "assigned_fraction": float((dist <= tolerance).mean()),
        "median_nn_distance": float(np.median(dist)),
        "p95_nn_distance": float(np.percentile(dist, 95)),
        "tolerance": tolerance,
    }
