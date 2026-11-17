"""Tria la finestra del Savitzky-Golay comú (adapter.FINESTRA).

Regla: la finestra senar més petita (ordre 2) que deixa el residu d'alta freqüència dels jugadors
d'IDSSE degradat (llavor 0) dins del ±30 % del residu objectiu de bench/target_stats.json. Si cap
finestra fins a 25 hi arriba, es fa servir 25. Només fa servir agregats del partit (el residu
objectiu), mai cap mètrica de possessió.

    /workspace/shared/venvs/bench/bin/python bench/tria_finestra.py
"""
import glob
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import adapter  # noqa: E402
import tracking_stats as ts  # noqa: E402

TOL = 0.30
MAX_W = 25


def residus(fitxers, w):
    rj, rb = [], []
    for f in fitxers:
        z = np.load(f, allow_pickle=False)
        xy = z["players_xy"].astype(np.float64)
        b = z["ball_xyz"][:, :2].astype(np.float64)
        mask = z["period"] > 0
        if w:
            xy = np.stack([np.stack([adapter.per_fragment(xy[:, s, c], w) for c in (0, 1)], axis=-1)
                           for s in range(xy.shape[1])], axis=1)
            b = np.stack([adapter.per_fragment(b[:, c], w) for c in (0, 1)], axis=-1)
        st = ts.stats_soroll(b, xy, mask)
        rj.append((st["jugadors_std_m"], st["n_residus_jugadors"]))
        rb.append((st["pilota_std_m"], st["n_residus_pilota"]))
    # desviació conjunta ponderada pel nombre de residus
    comb = lambda r: float(np.sqrt(sum(s * s * n for s, n in r) / sum(n for _, n in r)))
    return comb(rj), comb(rb)


def main():
    t = json.load(open(os.path.join(AQUI, "target_stats.json")))["soroll"]
    obj_j, obj_b = t["jugadors_std_m"], t["pilota_std_m"]
    fitxers = sorted(glob.glob("/workspace/shared/bench_v7_deg/idsse/*_s0.npz"))
    files, triada = [], None
    for w in [0] + list(range(5, MAX_W + 1, 2)):
        rj, rb = residus(fitxers, w)
        dins = abs(rj / obj_j - 1) <= TOL
        files.append({"finestra": w, "residu_jugadors_m": rj, "residu_pilota_m": rb, "dins_tolerancia": dins})
        print(f"finestra {w:>2}: jugadors {rj * 1000:.3f} mm (objectiu {obj_j * 1000:.3f}), "
              f"pilota {rb * 1000:.3f} mm (objectiu {obj_b * 1000:.3f}) {'<- dins' if dins else ''}", flush=True)
        if w and dins and triada is None:
            triada = w
            break
    out = {"objectiu": {"jugadors_m": obj_j, "pilota_m": obj_b}, "tolerancia": TOL, "files": files,
           "triada": triada or MAX_W, "arriba": triada is not None}
    with open(os.path.join(AQUI, "finestra_sg.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("triada:", out["triada"], "(dins de la tolerància)" if out["arriba"] else "(cap finestra hi arriba; 25)")


if __name__ == "__main__":
    main()
