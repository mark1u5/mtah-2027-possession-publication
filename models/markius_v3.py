#!/usr/bin/env python3
"""Markius v3 Entry Point: Autonomous Ballistic-Kinematic Perception Engine.

Synthesizes closed-form aerodynamic drag reconstruction, permutation-invariant pointwise pitch control,
multi-cue hybrid dead-ball detection, and ballistic flight-phase midpoint retiming.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from starter.markius_v3 import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Markius v3: Autonomous Ballistic-Kinematic Perception Engine")
    parser.add_argument("--game", default="BAR-ATH", help="Game key (default: BAR-ATH)")
    parser.add_argument("--out", default="evaluation/results/markius_v3.csv", help="Output submission CSV path")
    parser.add_argument("--data-dir", default=None, help="Path to data directory")
    parser.add_argument("--dwell-filter", action="store_true", help="Enable experimental 1.0s dwell filter")
    args = parser.parse_args()

    run_pipeline(game=args.game, out_csv=args.out, data_dir=args.data_dir, enable_dwell_filter=args.dwell_filter)
