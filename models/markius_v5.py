#!/usr/bin/env python3
"""Command-line entry point for Markius v5 (Hybrid TCN + Physical FSM)."""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from starter.markius_v5 import run_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Markius v5 hybrid possession tracker")
    parser.add_argument("--game", default="BAR-ATH", help="Game key (default: BAR-ATH)")
    parser.add_argument(
        "--out", default="evaluation/results/markius_v5.csv", help="Output submission CSV path"
    )
    parser.add_argument("--data-dir", default=None, help="Path to data directory")
    parser.add_argument(
        "--weights", default="cache/markius_v5_tcn_weights.npz", help="Path to TCN frozen weights archive"
    )
    args = parser.parse_args()
    run_pipeline(
        game=args.game,
        out_csv=args.out,
        data_dir=args.data_dir,
        tcn_weights_path=args.weights,
    )
