"""tlseval — evaluation and analysis for TLS individual-tree segmentation.

Quick start:

    from tlseval import evaluate, summarise
    df = evaluate("plot.laz")
    print(summarise(df))

The command line is the usual entry point:

    tlseval score plot.laz
    tlseval batch predictions/ --reference reference/
    tlseval report results/ --out report/
"""

from .core import (
    DEFAULTS,
    DETECTION_IOU_THRESHOLD,
    INSIDE_FIELDS,
    RESULT_COLUMNS,
    build_instances,
    classify_failures,
    compute_iou,
    config_from,
    encode_voxels,
    evaluate,
    match_instances,
    read_results,
    summarise,
    write_results,
)

__version__ = "0.2.0"

__all__ = [
    "DEFAULTS", "DETECTION_IOU_THRESHOLD", "INSIDE_FIELDS", "RESULT_COLUMNS",
    "build_instances", "classify_failures", "compute_iou", "config_from",
    "encode_voxels", "evaluate", "match_instances", "read_results",
    "summarise", "write_results", "__version__",
]
