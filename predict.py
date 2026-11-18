"""ATD tracking of BAR-ATH -> per-frame possession CSV (frame,label). Single inference script for v9m1.

    python predict.py --data /path/to/data --out submission.csv

--data is the folder of the competition data bundle containing tactical/BAR-ATH_tactical_atd.{txt,xml}
(default: $MTAH_DATA_DIR, else ./data). No label or Smart Tagging file is read.

Pipeline (v9m1):
  1. Kinematic features (base/bench/features.py) -> TCN v7d per-frame H/A/D probabilities
  2. Physical rules without Smart Tagging (principled_v1, p3_notag) + kinematic contact confidence
  3. Dynamic modulation w_t = w * (1 - factor * conf_t) + decoupled minimum-duration Viterbi HMM
"""
import argparse
import hashlib
import json
import os
import sys

# Fix CPU threads and disable GPU before importing torch/numpy for deterministic reproduction.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
_DEFAULT_THREADS = os.environ.get("MTAH_THREADS", "2")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, _DEFAULT_THREADS)

HERE = os.path.dirname(os.path.abspath(__file__))
_IN_REPO = os.path.join(HERE, "base")
BASE = os.environ.get("ULISES_BASE") or (_IN_REPO if os.path.isdir(_IN_REPO) else os.path.dirname(HERE))


def _repo_root(d):
    """Folder whose data/ is the default data location: MTAH_ROOT, else first parent with CHALLENGE.md."""
    if os.environ.get("MTAH_ROOT"):
        return os.environ["MTAH_ROOT"]
    while True:
        if os.path.exists(os.path.join(d, "CHALLENGE.md")):
            return d
        if os.path.dirname(d) == d:
            return HERE
        d = os.path.dirname(d)


REPO = _repo_root(HERE)
os.environ.setdefault("MTAH_DEG", "v3")
os.environ.setdefault("MTAH_DATA_DIR", os.path.join(REPO, "data"))
for _p in (HERE, BASE, os.path.join(BASE, "bench"), os.path.join(BASE, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def run_v9m1(d, tcn_path, params, threads=2):
    """Run v9m1 inference on a loaded match dict and return (labels, intermediates)."""
    import features as ft
    import tune_decoder as td
    from models import decoder as dc, tcn_v7

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)

    regles = params.get("regles", "p3_notag")
    model, ck = tcn_v7.carrega(tcn_path)
    X, _, _ = ft.features(d)
    p_tcn = tcn_v7.probabilitats(model, ck, X)

    labels_rules, clar = td.janv1_amb_control(d, regles)
    conf = td.conf_fisica(d) if "factor" in params["cfg"] else None
    r = (np.isfinite(d["bx"]) & np.isfinite(d["by"])) if params["cfg"].get("r") else None

    out = dc.descodifica(
        p_tcn,
        labels_rules,
        ~clar,
        np.asarray(d["period"]),
        params["st"],
        [params["cfg"]],
        conf=conf,
        r=r,
    )[0]

    return out, {
        "p_tcn": p_tcn,
        "rules": labels_rules,
        "clar": clar,
        "conf": conf,
        "period": np.asarray(d["period"]),
    }


def main():
    ap = argparse.ArgumentParser(description="Run v9m1 possession inference on ATD tracking.")
    ap.add_argument("--model", default=os.path.join("models", "final"), help="Folder with tcn.pt and params.json")
    ap.add_argument("--data", default=None, help="Data bundle folder containing tactical/ (default: $MTAH_DATA_DIR or ./data)")
    ap.add_argument("--out", default=None, help="Output CSV path (default: out/<model>.csv)")
    ap.add_argument("--threads", type=int, default=int(_DEFAULT_THREADS), help="CPU threads for PyTorch (default: 2)")
    ap.add_argument("--save-signals", help="Optional .npz path for intermediate signals (diagnostic only)")
    a = ap.parse_args()

    if a.data:
        os.environ["MTAH_DATA_DIR"] = os.path.abspath(a.data)
    mdir = a.model if os.path.isabs(a.model) else os.path.join(HERE, a.model)
    out_path = os.path.abspath(a.out or os.path.join(HERE, "out", f"{os.path.basename(mdir)}.csv"))
    with open(os.path.join(mdir, "params.json"), encoding="utf-8") as fh:
        params = json.load(fh)

    cwd = os.getcwd()
    os.chdir(BASE)
    try:
        import adapter
        d = adapter.carrega_partit()
        labels, sig = run_v9m1(d, os.path.join(mdir, "tcn.pt"), params, threads=a.threads)
    finally:
        os.chdir(cwd)

    if a.save_signals:
        np.savez_compressed(
            a.save_signals,
            p_tcn=sig["p_tcn"],
            rules=sig["rules"],
            clar=sig["clar"],
            conf=sig["conf"],
            y=labels,
            period=sig["period"],
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{lbl}\n" for i, lbl in enumerate(labels))

    with open(out_path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    with open(out_path + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{sha}  {os.path.basename(out_path)}\n")

    in_play = sig["period"] > 0
    lab_arr = np.asarray(labels)[in_play]
    res = {
        "csv": out_path,
        "sha256": sha,
        "frames": int(len(labels)),
        "pct": {c: round(100.0 * float(np.mean(lab_arr == c)), 2) for c in ("H", "A", "D")},
        "sha_p_tcn_5dec": hashlib.sha256(np.round(sig["p_tcn"], 5).tobytes()).hexdigest()[:16],
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
