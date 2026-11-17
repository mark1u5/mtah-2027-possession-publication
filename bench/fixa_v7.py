"""Part 1 del candidat v7: selecció pas a pas amb la graella "vora" i paràmetres finals.

    $P bench/fixa_v7.py bench/results/<resultat>.json

Aplica la selecció de docs/criteri_seleccio.md contra janv1_vit. Si la triada és porta_v7b (amb la
graella vora), desa a models/v7_params.json els paràmetres ajustats amb els 7 d'IDSSE i les
estadístiques de l'HMM dels 27 partits d'entrenament, i surt amb 0. Si no, no desa res i surt amb 3.
"""
import json
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import seleccio_pas as sp  # noqa: E402
import tune_decoder as td  # noqa: E402

CANDIDATS = ["porta_v7b_vora", "fusio_v7b", "tcn_vit_v7b_vora"]
ESPERADA = "porta_v7b_vora"
SORTIDA = os.path.join(os.path.dirname(AQUI), "models", "v7_params.json")


def main():
    with open(sys.argv[1]) as fh:
        res = json.load(fh)
    files, tria, motiu = sp.selecciona(res, "janv1_vit", CANDIDATS)
    for f in files:
        print(f"{f['model']:18s} dif {f['dif']:+.4f} [{f['ic95'][0]:+.4f}, {f['ic95'][1]:+.4f}] {f['guanya_partits']}/10 "
              f"(a) {'sí' if f['a'] else 'no'}  poss. {f['possessions']} (b) {'sí' if f['b'] else 'no'}")
    print(f"triada: {tria} ({motiu})")
    if tria != ESPERADA:
        print("la triada canvia: no es fixen paràmetres")
        sys.exit(3)
    p = td.ajusta("porta", set(td.plecs()["tots"][1]), "v7b", "vora")
    out = {"model": "v7", "descodificador": "porta", "tcn": "v7b", "graella": "vora",
           "cfg": p["cfg"], "spellF1_ajust_7_idsse": round(p["spellF1_ajust"], 6), "st": p["st"],
           "origen": os.path.basename(sys.argv[1])}
    with open(SORTIDA, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"paràmetres fixats: {p['cfg']} -> {os.path.relpath(SORTIDA, os.path.dirname(AQUI))}")


if __name__ == "__main__":
    main()
