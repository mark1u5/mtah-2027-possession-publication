"""Part 3 de la C2c: selecció pas a pas amb la v7d i paràmetres finals del candidat v7d.

    $P bench/fixa_v7d.py bench/results/<resultat>.json

Aplica la selecció de docs/criteri_seleccio.md contra janv1_vit amb porta_v7d, fusio_v7d i
tcn_vit_v7d. Informa també, sense que decideixi, de la diferència a Metrica de cada configuració i
de la v7 actual (porta_v7b_vora) contra janv1_vit, amb l'IC del bootstrap aparellat dels 3 partits.
Desa bench/results/c2c_seleccio.json i .md. Si la triada és porta_v7d, desa models/v7d_params.json
(paràmetres ajustats amb els 7 d'IDSSE i estadístiques dels 27) i surt amb 0; si no, surt amb 3.
"""
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import seleccio_pas as sp  # noqa: E402
import tune_decoder as td  # noqa: E402

REF = "janv1_vit"
ACTUAL = "porta_v7b_vora"
CANDIDATS = ["porta_v7d", "fusio_v7d", "tcn_vit_v7d"]
ESPERADA = "porta_v7d"
N_BOOT, LLAVOR_BOOT = 2000, 0  # els de l'arnès
SORTIDA = os.path.join(os.path.dirname(AQUI), "models", "v7d_params.json")


def boot(v):
    v = np.asarray(v, float)
    rng = np.random.RandomState(LLAVOR_BOOT)
    m = v[rng.randint(0, len(v), (N_BOOT, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def dif_metrica(res, m):
    pa, pb = res["principal"][REF]["per_partit"], res["principal"][m]["per_partit"]
    ks = sorted(k for k in pa if k.startswith("metrica/"))
    return boot([pb[k]["spellF1"] - pa[k]["spellF1"] for k in ks])


def main():
    with open(sys.argv[1]) as fh:
        res = json.load(fh)
    files, tria, motiu = sp.selecciona(res, REF, CANDIDATS)
    taula = []
    for m in [REF, ACTUAL] + CANDIDATS:
        p, s = res["principal"][m], res["partit"][m]["sanitat"]
        f = {"model": m, "spellF1_10": p["spellF1"], "idsse": p["spellF1_idsse"], "metrica": p["spellF1_metrica"],
             "possessions": s["possessions"], "domini": bool(s["domini"])}
        if m != REF:
            dif, ic, guanya = sp.comparacio(res, REF, m)
            f.update({"dif_10": [dif] + list(ic), "guanya_partits": guanya, "dif_metrica": list(dif_metrica(res, m))})
        taula.append(f)
    L = ["| configuració | spellF1 10 [IC95] | IDSSE | Metrica | dif. 10 contra janv1_vit [IC95] | dif. Metrica [IC95, 3 partits] | poss. partit | domini |",
         "|---|---|---|---|---|---|---|---|"]
    for f in taula:
        a = "{:.3f} [{:.3f}; {:.3f}]".format(*f["spellF1_10"])
        d10 = "{:+.3f} [{:+.3f}; {:+.3f}] ({}/10)".format(*f["dif_10"], f["guanya_partits"]) if "dif_10" in f else "–"
        dm = "{:+.3f} [{:+.3f}; {:+.3f}]".format(*f["dif_metrica"]) if "dif_metrica" in f else "–"
        L.append(f"| {f['model']} | {a} | {f['idsse']:.3f} | {f['metrica']:.3f} | {d10} | {dm} | {f['possessions']} | "
                 f"{'dins' if f['domini'] else 'fora'} |")
    L += ["", f"Triada: {tria} ({motiu})."]
    # porta_v7d contra la v7 actual, informatiu
    try:
        dif, ic, g = sp.comparacio(res, ACTUAL, "porta_v7d")
        L.append(f"porta_v7d menys {ACTUAL}: {dif:+.3f} [{ic[0]:+.3f}; {ic[1]:+.3f}], {g}/10 partits.")
    except KeyError:
        pass
    print("\n".join(L))
    out = {"resultat": os.path.basename(sys.argv[1]), "taula": taula, "files_seleccio": files, "triada": tria, "motiu": motiu}
    if tria == ESPERADA:
        p = td.ajusta("porta", set(td.plecs()["tots"][1]), "v7d", "c2c")
        params = {"model": "v7d", "descodificador": "porta", "tcn": "v7d", "graella": "c2c", "degradacio": "v3",
                  "cfg": p["cfg"], "spellF1_ajust_7_idsse": round(p["spellF1_ajust"], 6), "st": p["st"],
                  "origen": os.path.basename(sys.argv[1])}
        with open(SORTIDA, "w") as fh:
            json.dump(params, fh, indent=1, sort_keys=True)
            fh.write("\n")
        out["cfg_final"] = p["cfg"]
        print(f"paràmetres fixats: {p['cfg']} -> models/v7d_params.json")
    with open(os.path.join(AQUI, "results", "c2c_seleccio.json"), "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    with open(os.path.join(AQUI, "results", "c2c_seleccio.md"), "w") as fh:
        fh.write("# Selecció C2c (v7d)\n\n" + "\n".join(L) + "\n")
    if tria != ESPERADA:
        print("la triada no és porta_v7d: no es fixen paràmetres")
        sys.exit(3)


if __name__ == "__main__":
    main()
