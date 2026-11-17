"""Diferència de domini de les característiques: partit contra IDSSE i SkillCorner degradats.

Només agregats: mitjana, desviació i quantils p5, p50 i p95 de cada característica, calculats als
frames on és vàlida (segons el seu indicador, vegeu features.VALIDESA), i la diferència de mitjanes
estandarditzada (partit menys extern, dividit per la desviació conjunta). Es marquen les
característiques en què el partit i IDSSE difereixen en més de 0,5 desviacions.

    /workspace/shared/venvs/bench/bin/python bench/feature_gap.py [--seed 0]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
ARREL = os.path.dirname(AQUI)
sys.path.insert(0, ARREL)
sys.path.insert(0, AQUI)
import features as ft  # noqa: E402

LLINDAR = 0.5


def stats(vals):
    if vals.size == 0:
        return None
    p5, p50, p95 = np.percentile(vals, [5, 50, 95])
    return {"mitjana": float(vals.mean()), "desv": float(vals.std()), "p5": float(p5), "p50": float(p50),
            "p95": float(p95), "n": int(vals.size)}


def per_caracteristica(blocs):
    """blocs: llista de (X, noms, màscara de frames). Agrega els frames de tots els partits."""
    noms = blocs[0][1]
    out = {}
    for j, k in enumerate(noms):
        vals = []
        for X, nn, m in blocs:
            mm = m.copy()
            for ind in ft.VALIDESA.get(k, []):
                mm &= X[:, nn.index(ind)] > 0.5
            vals.append(X[mm, j])
        out[k] = stats(np.concatenate(vals).astype(np.float64))
    return out


def dif(a, b):
    if not a or not b:
        return None
    s = np.sqrt((a["desv"] ** 2 + b["desv"] ** 2) / 2)
    return float((a["mitjana"] - b["mitjana"]) / s) if s > 0 else 0.0


def carrega_ext(font, seed):
    blocs = []
    for f in sorted(glob.glob(os.path.join(ft.SORTIDA_EXT, font, f"*_s{seed}.npz"))):
        z = np.load(f, allow_pickle=False)
        blocs.append((z["X"], [str(x) for x in z["noms"]], z["period"] > 0))
    return blocs


def carrega_partit():
    import games
    z = np.load(os.path.join(ft.SORTIDA_PARTIT, f"{games.GAME}.npz"), allow_pickle=False)
    X = z["X"]
    m = np.zeros(len(X), bool)
    for _, a, b in games.half_video_blocks(games.GAME):
        m[a:b + 1] = True
    return [(X, [str(x) for x in z["noms"]], m)]


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dom = {"partit": per_caracteristica(carrega_partit()),
           "idsse": per_caracteristica(carrega_ext("idsse", a.seed)),
           "skillcorner": per_caracteristica(carrega_ext("skillcorner", a.seed))}
    noms = list(dom["partit"])
    files, marcades = [], []
    for k in noms:
        di, ds = dif(dom["partit"][k], dom["idsse"][k]), dif(dom["partit"][k], dom["skillcorner"][k])
        if di is not None and abs(di) > LLINDAR:
            marcades.append((k, di))
        files.append((k, di, ds))
    L = ["# Diferència de domini de les característiques", "",
         f"Generat per `bench/feature_gap.py`. Partit: frames de les dues parts. IDSSE i SkillCorner: "
         f"degradats, llavor {a.seed}, frames de les parts. Estadístiques als frames on cada característica "
         "és vàlida (segons el seu indicador). d = (mitjana del partit − mitjana externa) / desviació "
         f"conjunta. Es marquen amb ⚠ les |d| > {LLINDAR} entre el partit i IDSSE. Només agregats.", "",
         "| característica | partit mitj. ± desv. [p5, p50, p95] | IDSSE mitj. ± desv. [p5, p50, p95] "
         "| SkillCorner mitj. ± desv. [p5, p50, p95] | d partit−IDSSE | d partit−SC |",
         "|---|---|---|---|---|---|"]
    cel = lambda s: "-" if not s else f"{s['mitjana']:.3f} ± {s['desv']:.3f} [{s['p5']:.2f}, {s['p50']:.2f}, {s['p95']:.2f}]"
    for k, di, ds in files:
        marca = " ⚠" if di is not None and abs(di) > LLINDAR else ""
        L.append(f"| {k}{marca} | {cel(dom['partit'][k])} | {cel(dom['idsse'][k])} | {cel(dom['skillcorner'][k])} "
                 f"| {f3(di)} | {f3(ds)} |")
    L += ["", f"Característiques amb |d| > {LLINDAR} entre el partit i IDSSE: "
          + (", ".join(f"`{k}` ({v:+.2f})" for k, v in marcades) if marcades else "cap") + "."]
    with open(os.path.join(AQUI, "report_feature_gap.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(AQUI, "report_feature_gap.json"), "w") as fh:
        json.dump({"llavor": a.seed, "dominis": dom, "marcades": marcades}, fh, indent=1, ensure_ascii=False)
    print("marcades (|d| > 0,5 partit−IDSSE):", ", ".join(f"{k} {v:+.2f}" for k, v in marcades) or "cap")


if __name__ == "__main__":
    main()
