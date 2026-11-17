"""Comprovació de domini (docs/criteri_seleccio.md).

El prior de possessions és la mitjana de possessions per partit de label_acc als 7 partits
d'IDSSE, escalada a la durada jugada del partit proporcionat. Un candidat només és vàlid si les
possessions que preveu al partit són entre 0,5 i 1,5 vegades aquest prior. Fa servir només el
tracking (durada de les parts, de target_stats.json) i les prediccions en agregat, mai la GT.

    /workspace/shared/venvs/bench/bin/python bench/domini.py     # imprimeix el prior i l'interval
"""
import glob
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(AQUI), "evaluation"))
import possession_metrics as pm  # noqa: E402

ORIGINALS = "/workspace/shared/bench_v7"   # etiquetes sense degradar (iguals a totes les degradacions)
FPS = 25
FACTORS = (0.5, 1.5)


def blocs(periode):
    out, a = [], None
    for i, p in enumerate(np.append(periode, 0)):
        if a is not None and p != periode[a]:
            out.append((a, i))
            a = None
        if a is None and i < len(periode) and p > 0:
            a = i
    return out


def prior():
    poss, mins = [], []
    for f in sorted(glob.glob(os.path.join(ORIGINALS, "idsse", "*.npz"))):
        z = np.load(f, allow_pickle=False)
        runs = blocs(z["period"])
        codes = pm.label_codes(z["label_acc"].astype("<U1"))
        poss.append(pm.possession_stats(pm.fill_abstentions(codes, runs), runs)["n_spells"])
        mins.append(sum(b - a for a, b in runs) / FPS / 60)
    with open(os.path.join(AQUI, "target_stats.json")) as fh:
        min_partit = json.load(fh)["frames"] / FPS / 60
    p = float(np.mean(poss)) * min_partit / float(np.mean(mins))
    return {"prior": round(p, 1), "interval": [round(FACTORS[0] * p, 1), round(FACTORS[1] * p, 1)],
            "possessions_idsse": poss, "minuts_idsse": [round(m, 2) for m in mins],
            "minuts_partit": round(min_partit, 2)}


def dins(n_possessions, pr=None):
    pr = pr or prior()
    lo, hi = pr["interval"]
    return lo <= n_possessions <= hi


if __name__ == "__main__":
    print(json.dumps(prior(), indent=1))
