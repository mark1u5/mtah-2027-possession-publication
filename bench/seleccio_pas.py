"""Selecció pas a pas (C2b) sobre un resultat de l'arnès.

Una configuració entra si (a) l'IC95 aparellat de la diferència de spellF1 contra la referència és
> 0 sobre els 10 partits i (b) passa la comprovació de domini al partit. Si n'hi ha més d'una, es
queda la de diferència més gran; si el seu IC se solapa amb el d'una més simple que també compleix,
es queda la més simple. L'ordre de simplicitat és el de --candidats.

    $P bench/seleccio_pas.py bench/results/<resultat>.json --referencia janv1_vit \
        --candidats porta_v7 porta_v7b fusio_v7 fusio_v7b tcn_vit_v7 tcn_vit_v7b
"""
import argparse
import json


def comparacio(res, ref, m):
    for c in res["comparacions"]:
        if (c["a"], c["b"]) == (ref, m):
            return c["dif_spellF1_b_menys_a"], c["ic95"], c["b_millor_en_partits"]
        if (c["a"], c["b"]) == (m, ref):
            return -c["dif_spellF1_b_menys_a"], [-c["ic95"][1], -c["ic95"][0]], c["partits"] - c["b_millor_en_partits"]
    raise KeyError(f"cap comparació entre {ref} i {m}")


def selecciona(res, ref, candidats):
    files, valids = [], []
    for i, m in enumerate(candidats):
        dif, ic, guanya = comparacio(res, ref, m)
        san = res["partit"][m]["sanitat"]
        f = {"model": m, "dif": dif, "ic95": ic, "guanya_partits": guanya, "a": ic[0] > 0,
             "possessions": san["possessions"], "b": bool(san["domini"]), "ordre": i}
        files.append(f)
        if f["a"] and f["b"]:
            valids.append(f)
    if not valids:
        return files, ref, "cap candidat no compleix (a) i (b): es queda la referència"
    lider = max(valids, key=lambda f: f["dif"])
    solapen = [f for f in valids if f["ic95"][1] >= lider["ic95"][0] and lider["ic95"][1] >= f["ic95"][0]]
    tria = min(solapen, key=lambda f: f["ordre"])
    motiu = f"diferència més gran: {lider['model']}"
    if tria is not lider:
        motiu += f"; el seu IC se solapa amb el de {tria['model']}, que és més simple"
    return files, tria["model"], motiu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("resultat")
    ap.add_argument("--referencia", default="janv1_vit")
    ap.add_argument("--candidats", nargs="+", required=True)
    a = ap.parse_args()
    with open(a.resultat) as fh:
        res = json.load(fh)
    files, tria, motiu = selecciona(res, a.referencia, a.candidats)
    print(f"prior de domini: {res['partit']['_prior_domini']['prior']}, interval {res['partit']['_prior_domini']['interval']}")
    for f in files:
        print(f"{f['model']:14s} dif {f['dif']:+.4f} [{f['ic95'][0]:+.4f}, {f['ic95'][1]:+.4f}] {f['guanya_partits']}/10 "
              f"(a) {'sí' if f['a'] else 'no'}  poss. {f['possessions']} (b) {'sí' if f['b'] else 'no'}")
    print(f"triada: {tria} ({motiu})")


if __name__ == "__main__":
    main()
