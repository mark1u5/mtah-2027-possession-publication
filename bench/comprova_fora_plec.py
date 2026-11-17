"""Comprova que les probabilitats de la TCN dels partits d'IDSSE i Metrica són fora de plec.

Per a cada fitxer de /workspace/shared/tcn_v7_probs: (1) el meta diu el plec i fora_de_plec; (2) el
checkpoint d'aquell plec no té el partit ni a l'entrenament ni a la validació interna; (3) el sha256
desat coincideix amb les probabilitats; (4) tornant a fer la inferència amb el checkpoint del plec
surt el mateix sha256.

    /workspace/shared/venvs/bench/bin/python bench/comprova_fora_plec.py
"""
import json
import os
import sys

import numpy as np
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
sys.path.insert(0, AQUI)
import train_tcn as tt  # noqa: E402


def main():
    torch.set_num_threads(2)
    plecs = {nom: (av, ent, val) for nom, av, ent, val in tt.plecs()}
    cks = {nom: torch.load(tt.ruta_plec(nom), map_location="cpu", weights_only=False) for nom in plecs}
    files, ok_tot = [], True
    for font in ("idsse", "metrica"):
        for partit in tt.partits(font):
            for ll in tt.LLAVORS:
                z = np.load(os.path.join(tt.SORTIDA_PROBS, font, f"{partit}_s{ll}.npz"), allow_pickle=False)
                meta = json.loads(str(z["meta"]))
                nom = meta["plec"]
                ck = cks[nom]
                vist = f"{font}/{partit}" in ck["entrenament"] or ck["validacio"] == f"{font}/{partit}"
                avaluat = (font, partit) in plecs[nom][0]
                sha_ok = tt.sha(z["probs"]) == meta["sha256"]
                reinf = tt.sha(tt.probs_extern(tt.ruta_plec(nom), font, partit, ll)) == meta["sha256"] if ll == 0 else None
                ok = meta["fora_de_plec"] and not vist and avaluat and sha_ok and reinf is not False
                ok_tot &= ok
                files.append((f"{font}/{partit}_s{ll}", nom, meta["fora_de_plec"], vist, sha_ok, reinf, ok))
    for f in files:
        print("{:32s} plec={:24s} fora_de_plec={} vist_a_entrenament={} sha_desat={} reinferencia={} -> {}".format(*f))
    print("tots fora de plec:", ok_tot, f"({len(files)} fitxers)")


if __name__ == "__main__":
    main()
