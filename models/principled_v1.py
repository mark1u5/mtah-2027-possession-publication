#!/usr/bin/env python3
"""Principled v1 entry point. See starter/principled_v1.py."""

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from starter.principled_v1 import run_pipeline  # noqa: E402,F401

if __name__ == "__main__":
    runpy.run_path(os.path.join(ROOT, "starter", "principled_v1.py"), run_name="__main__")
