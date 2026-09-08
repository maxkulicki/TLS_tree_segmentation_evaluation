#!/usr/bin/env python3
"""Backwards-compatible shim.

The scorer moved into the `tlseval` package when batch mode and the analysis
tools were added. This file keeps `python evaluate.py plot.laz` working, and
keeps `from evaluate import evaluate, read_results` importable for anyone who
wrote against the previous single-file layout.

Prefer the command line:  tlseval score plot.laz
"""
import sys

from tlseval.core import *            # noqa: F401,F403
from tlseval.core import evaluate, read_results, write_results  # noqa: F401


def main():
    from tlseval.cli import main as _main
    return _main(["score"] + sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
