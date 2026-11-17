#!/usr/bin/env python3
"""Check that every model entry point imports and exposes run_pipeline.

Needs no match data. Run from anywhere:

    python tests/smoke_imports.py

Each script is executed in a fresh interpreter from a neutral working directory, the same way
the README tells people to run it, so a wrong sys.path or a missing module shows up here and not
on the day of the submission.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENTRY_POINTS = [
    "models/markius_v1.py",
    "models/markius_v2.py",
    "models/markius_v3.py",
    "models/markius_v4.py",
    "models/markius_v5.py",
    "models/markius_v6.py",
    "models/principled_v1.py",
    "starter/principled_v1.py",
    "models/deprecated/el_chato_model_v6.py",
    "models/deprecated/el_chato_model_v7.py",
    "models/deprecated/el_chato_model_v8.py",
    "models/deprecated/el_chato_model_v9.py",
    "models/deprecated/el_chato_model_v10.py",
    "starter/markius_v1.py",
    "starter/markius_v2.py",
    "starter/markius_v3.py",
    "starter/markius_v4.py",
    "starter/markius_v5.py",
    "starter/markius_v6.py",
]

# Load the file as a module without triggering its __main__ block, then look for run_pipeline.
# For the thin entry points in models/, run_pipeline is the name they import from starter/.
PROBE = r"""
import importlib.util, sys
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("probe", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
if not callable(getattr(mod, "run_pipeline", None)):
    sys.exit("no run_pipeline")
"""


def main():
    failures = 0
    for rel in ENTRY_POINTS:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"MISSING  {rel}")
            failures += 1
            continue
        proc = subprocess.run([sys.executable, "-c", PROBE, path], cwd="/",
                              capture_output=True, text=True)
        if proc.returncode == 0:
            print(f"ok       {rel}")
        else:
            last = (proc.stderr.strip().splitlines() or ["?"])[-1]
            print(f"FAIL     {rel}: {last}")
            failures += 1
    print(f"\n{len(ENTRY_POINTS) - failures}/{len(ENTRY_POINTS)} ok")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
