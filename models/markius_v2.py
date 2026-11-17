#!/usr/bin/env python3
"""Markius v2 entry point.

Execute Markius v2 autonomous multi-cue possession prediction pipeline.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from starter.markius_v2 import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Markius v2: Advanced Autonomous Multi-Cue Possession Engine")
    parser.add_argument("--game", default="BAR-ATH", help="Game key (default: BAR-ATH)")
    parser.add_argument("--out", default="evaluation/results/markius_v2.csv", help="Output submission CSV path")
    parser.add_argument("--data-dir", default=None, help="Path to data directory")
    args = parser.parse_args()

    run_pipeline(game=args.game, out_csv=args.out, data_dir=args.data_dir)
