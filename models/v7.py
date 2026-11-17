"""Candidat v7: porta amb la TCN v7b sobre les regles janv1 p3 (amb Smart Tagging).

Igual que porta_v7b del banc (docs/descodificador.md), però la capa de regles és janv1 p3 en lloc
de janv1_notag, i el control clar que fa de porta també surt de p3. Els paràmetres del
descodificador i les estadístiques de l'HMM són els del banc (models/v7_params.json). Vegeu
docs/v7.md.
"""
import json
import os
import sys

import numpy as np

ARREL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PESOS = os.path.join(ARREL, "models", "tcn_v7b_weights.pt")
PARAMS = os.path.join(ARREL, "models", "v7_params.json")
# v7d (C2c): el mateix muntatge amb la TCN v7d i els seus paràmetres
CANDIDATS = {"v7": ("v7b", PARAMS), "v7d": ("v7d", os.path.join(ARREL, "models", "v7d_params.json"))}


def pesos(tcn):
    return os.path.join(ARREL, "models", f"tcn_{tcn}_weights.pt")
FILS = 2  # els mateixos que a l'entrenament


def regles(d, config="p3"):
    """Etiquetes de janv1 i el seu control clar. p3 necessita el Smart Tagging del partit."""
    from starter import principled_v1 as pv
    cfg = {**pv.CONFIGS[config], "exposa_control": True}
    kin = pv.kinematics_from_arrays(d["px"].copy(), d["py"].copy(), d["mean_speed"].copy(),
                                    d["bx"].copy(), d["by"].copy(), d["team_of_track"], d["pitch"])
    st = {"home": None, "away": None, "set_pieces": None}
    if cfg["visible_dead"] == "st_veto" or cfg["init"] == "v1":
        if "smart_tagging" not in d:
            raise RuntimeError(f"{config} necessita Smart Tagging")
        st["home"], st["away"], st["set_pieces"] = pv.load_smart_tagging_phases(d["smart_tagging"], d["n_frames"])
    labels, info = pv.predict(kin, st, d["n_frames"], d["half_blocks"], d["kickoff_2_frame"], d["pitch"], cfg)
    return labels, info["control_clar"]


def probs_tcn(d, pesos=PESOS, fils=FILS):
    import torch
    sys.path.insert(0, os.path.join(ARREL, "bench"))
    import features as ft
    from models import tcn_v7
    torch.set_num_threads(fils)
    torch.use_deterministic_algorithms(True)
    model, ck = tcn_v7.carrega(pesos)
    return tcn_v7.probabilitats(model, ck, ft.features(d)[0])


def carrega_params(path=PARAMS):
    with open(path) as fh:
        return json.load(fh)


def prediu(d, params=None, pesos=PESOS, config="p3", decimals=None):
    """Etiquetes del candidat i el seu control clar. decimals arrodoneix les probabilitats abans
    del descodificador (només si cal per a la reproductibilitat)."""
    from models import decoder as dc
    params = params or carrega_params()
    labels, clar = regles(d, config)
    p = probs_tcn(d, pesos)
    if decimals is not None:
        p = np.round(p, decimals)
    out = dc.descodifica(p, labels, ~clar, np.asarray(d["period"]), params["st"], [params["cfg"]])[0]
    return out, clar
