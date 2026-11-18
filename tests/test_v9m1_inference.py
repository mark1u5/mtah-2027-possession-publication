#!/usr/bin/env python3
"""End-to-end synthetic smoke test for the v9m1 inference pipeline (no match data needed).

    python tests/test_v9m1_inference.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "base"))
sys.path.insert(0, os.path.join(ROOT, "base", "bench"))

from predict import run_v9m1  # noqa: E402


def make_synthetic_match(n_frames=1600, seed=7):
    rng = np.random.default_rng(seed)
    n_tracks = 22
    px = np.full((n_frames, n_tracks), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_tracks), np.nan, dtype=np.float32)
    team_of_track = np.array([0] * 11 + [1] * 11, dtype=np.int8)

    # Spread 11 home and 11 away players smoothly across a 105x68 pitch (0..105, 0..68).
    base_x = np.concatenate([np.linspace(15, 65, 11), np.linspace(40, 90, 11)])
    base_y = np.tile(np.linspace(10, 58, 11), 2)
    t = np.arange(n_frames, dtype=np.float32)
    for k in range(n_tracks):
        px[:, k] = base_x[k] + 2.5 * np.sin(0.03 * t + k)
        py[:, k] = base_y[k] + 2.0 * np.cos(0.025 * t + 0.5 * k)

    # Move ball near home player #5 in half 1, then near away player #16 in half 2.
    bx = np.empty(n_frames, dtype=np.float32)
    by = np.empty(n_frames, dtype=np.float32)
    half = n_frames // 2
    bx[:half] = px[:half, 5] + 0.3
    by[:half] = py[:half, 5] + 0.2
    bx[half:] = px[half:, 16] - 0.3
    by[half:] = py[half:, 16] - 0.2
    # Inject a short ball tracking dropout.
    bx[300:340] = np.nan
    by[300:340] = np.nan

    period = np.ones(n_frames, dtype=np.int8)
    period[half:] = 2
    _ = rng  # keep deterministic signature

    import adapter
    d = {
        "domini": "partit",
        "font": "synthetic",
        "partit": "synthetic_v9m1",
        "px": px,
        "py": py,
        "bx": bx,
        "by": by,
        "team_of_track": team_of_track,
        "period": period,
        "n_frames": n_frames,
        "half_blocks": [(1, 0, half - 1), (2, half, n_frames - 1)],
        "kickoff_2_frame": half,
        "xuts_inicials": {1: 0, 2: half},
        "pitch": (105.0, 68.0),
    }
    return adapter.cinematica(d)


def main():
    mdir = os.path.join(ROOT, "models", "final")
    with open(os.path.join(mdir, "params.json"), encoding="utf-8") as fh:
        params = json.load(fh)
    d = make_synthetic_match()
    labels_a, sig_a = run_v9m1(d, os.path.join(mdir, "tcn.pt"), params, threads=2)
    labels_b, _ = run_v9m1(make_synthetic_match(), os.path.join(mdir, "tcn.pt"), params, threads=2)

    assert len(labels_a) == d["n_frames"]
    assert set(np.unique(labels_a)).issubset({"H", "A", "D", "X"})
    assert np.array_equal(labels_a, labels_b), "v9m1 inference must be deterministic"
    assert sig_a["p_tcn"].shape == (d["n_frames"], 3)
    assert np.allclose(sig_a["p_tcn"].sum(axis=1), 1.0, atol=1e-4)
    assert sig_a["conf"].shape == (d["n_frames"],)
    print(f"OK: tests/test_v9m1_inference.py (deterministic {len(labels_a)} frames, counts={dict(zip(*np.unique(labels_a, return_counts=True)))})")


if __name__ == "__main__":
    main()
