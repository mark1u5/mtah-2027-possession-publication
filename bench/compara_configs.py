"""Compara dos resultats de l'arnès (mateix model, configuracions diferents) amb el bootstrap
aparellat per partits sobre els 10 de selecció. Fa servir el spellF1 mitjà per partit sobre les
llavors de cada resultat.

    python bench/compara_configs.py A.json B.json --model janv1_notag

Dona B − A. B entra només si l'IC95 queda sencer per sobre de 0.
"""
import argparse
import json
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from harness import SELECCIO, bootstrap_mitjana  # noqa: E402


def per_partit(path, model):
    with open(path) as fh:
        r = json.load(fh)
    pp = r["principal"][model]["per_partit"]
    return {k: v["spellF1"] for k, v in pp.items() if k.split("/")[0] in SELECCIO}, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--model", default="janv1_notag")
    a = ap.parse_args()
    pa, ra = per_partit(a.a, a.model)
    pb, rb = per_partit(a.b, a.model)
    claus = sorted(set(pa) & set(pb))
    dif = [pb[k] - pa[k] for k in claus]
    m, lo, hi = bootstrap_mitjana(dif)
    out = {"a": os.path.basename(a.a), "b": os.path.basename(a.b), "model": a.model, "partits": len(claus),
           "llavors": [ra["llavors"], rb["llavors"]],
           "spellF1_a": bootstrap_mitjana([pa[k] for k in claus]), "spellF1_b": bootstrap_mitjana([pb[k] for k in claus]),
           "dif_b_menys_a": m, "ic95": [lo, hi], "b_millor_en_partits": int(sum(x > 0 for x in dif)),
           "b_guanya": lo > 0, "per_partit": {k: [pa[k], pb[k]] for k in claus}}
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
