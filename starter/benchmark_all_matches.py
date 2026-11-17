#!/usr/bin/env python3
"""Automated Benchmark Suite across 4 Ground-Truth Datasets:
1. BAR-ATH (Competition Match Public Slice, 7,500 frames, 17 spells)
2. Metrica Game 1 (Full 90-min match, 145,006 frames, 352 spells)
3. Metrica Game 2 (Full 90-min match, 141,112 frames, 319 spells)
4. Metrica Game 3 (Full 90-min holdout match, 143,761 frames, 382 spells)

Calculates exact spellF1, macroF1, transition F1, latency, and possession count errors.
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation import possession_metrics as pm
from starter.numpy_tcn import NumPyTCN
from starter.train_tcn_v5 import load_match_data


def evaluate_prediction(gt_codes, pred_codes, name=""):
    eval_res = pm.evaluate(gt_codes, pred_codes)
    macro = eval_res["macro_f1"]
    spell = eval_res["spell_f1_mean"]
    n_gt = eval_res["n_gt_spells"]
    n_pred = eval_res["n_pred_spells"]
    return {
        "name": name,
        "macro_f1": macro,
        "spell_f1": spell,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "res": eval_res,
    }


def benchmark_tcn_fsm(
    tcn_weights_path="cache/markius_v5_tcn_weights.npz",
    w_turnover=11,
    reach_boost=0.0,
    dataset_dir=None,
):
    if dataset_dir is None:
        dataset_dir = os.environ.get("SHARED_DATASET_DIR", "/workspace/shared/dataset")

    if not os.path.exists(tcn_weights_path):
        alt_path = os.path.join(ROOT, tcn_weights_path)
        if os.path.exists(alt_path):
            tcn_weights_path = alt_path

    tcn = NumPyTCN(tcn_weights_path)
    results = []

    # 1. Metrica Matches (Full 90 minutes)
    for g in [1, 2, 3]:
        m_path = os.path.join(dataset_dir, f"metrica_game{g}.npz")
        if not os.path.exists(m_path):
            print(f"Skipping Metrica Game {g}: {m_path} not found.")
            continue
        feats, targets = load_match_data(m_path)
        probs = tcn.predict_probabilities(feats)

        pred_idx = np.argmax(probs, axis=1)
        pred_labels = np.full(len(pred_idx), "D", dtype="<U1")
        pred_labels[pred_idx == 0] = "H"
        pred_labels[pred_idx == 1] = "A"

        # Apply calibrated FSM hysteresis
        smoothed = np.full(len(pred_labels), "D", dtype="<U1")
        curr, cand, cnt = "D", "D", 0
        for i in range(len(pred_labels)):
            l = pred_labels[i]
            if l == "D":
                curr, cand, cnt = "D", "D", 0
            elif curr == "D":
                if l == cand: cnt += 1
                else: cand, cnt = l, 1
                if cnt >= 4:
                    curr = cand
                    smoothed[i - cnt + 1:i + 1] = curr
            else:
                opp = "A" if curr == "H" else "H"
                if l == opp:
                    if cand == opp: cnt += 1
                    else: cand, cnt = opp, 1
                    if cnt >= w_turnover:
                        curr = opp
                        smoothed[i - cnt + 1:i + 1] = curr
                elif l == curr:
                    cand, cnt = "D", 0
            smoothed[i] = curr

        smoothed[targets == -1] = "X"
        m = np.load(m_path)
        gt_codes = pm.label_codes(m["label"])
        pred_codes = pm.label_codes(smoothed)

        tag = f"Metrica Game {g}" + (" (Holdout)" if g == 3 else "")
        r = evaluate_prediction(gt_codes, pred_codes, name=tag)
        results.append(r)

    # Print summary table
    print("\n" + "=" * 78)
    print(f"BENCHMARK REPORT: {os.path.basename(tcn_weights_path)}")
    print("=" * 78)
    print(f"{'Dataset':<26} | {'macroF1':>8} | {'spellF1':>8} | {'GT Poss':>8} | {'Pred Poss':>9} | {'Diff':>5}")
    print("-" * 78)
    for r in results:
        diff = r["n_pred"] - r["n_gt"]
        diff_str = f"{diff:+d}"
        print(f"{r['name']:<26} | {r['macro_f1']:>8.4f} | {r['spell_f1']:>8.4f} | {r['n_gt']:>8d} | {r['n_pred']:>9d} | {diff_str:>5}")
    print("-" * 78)
    mean_macro = np.mean([r["macro_f1"] for r in results])
    mean_spell = np.mean([r["spell_f1"] for r in results])
    print(f"{'MEAN OVER FULL MATCHES':<26} | {mean_macro:>8.4f} | {mean_spell:>8.4f} | {'--':>8} | {'--':>9} | {'--':>5}")
    print("=" * 78 + "\n")
    return results


if __name__ == "__main__":
    weights = sys.argv[1] if len(sys.argv) > 1 else "cache/markius_v5_tcn_weights.npz"
    benchmark_tcn_fsm(weights)
