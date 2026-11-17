#!/usr/bin/env python3
"""Markius v1 entry point.

Execute clean, legal possession prediction pipeline.
"""

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from starter.markius_v1 import run_pipeline
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Markius v1: Clean & Legal Possession Model")
    parser.add_argument("--game", default="BAR-ATH", help="Game key (default: BAR-ATH)")
    parser.add_argument("--out", default="evaluation/results/markius_v1.csv", help="Output submission CSV path")
    parser.add_argument("--data-dir", default=None, help="Path to data directory")
    args = parser.parse_args()

    run_pipeline(game=args.game, out_csv=args.out, data_dir=args.data_dir)
