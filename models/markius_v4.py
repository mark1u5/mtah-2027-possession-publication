#!/usr/bin/env python3
"""Command-line entry point for Markius v4."""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from starter.markius_v4 import run_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Markius v4 possession tracker")
    parser.add_argument("--game", default="BAR-ATH", help="Game key (default: BAR-ATH)")
    parser.add_argument(
        "--out", default="evaluation/results/markius_v4.csv", help="Output submission CSV path"
    )
    parser.add_argument("--data-dir", default=None, help="Path to data directory")
    args = parser.parse_args()
    run_pipeline(
        game=args.game,
        out_csv=args.out,
        data_dir=args.data_dir,
    )
