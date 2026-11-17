"""Markius v9: Descodificador HMM físic pur amb modulació dinàmica de TCN per contacte cinemàtic.

Integritat metodològica segons acords de la Fase 1:
- El descodificador no afegeix cap llindar de velocitat. Les etiquetes de janv1 que rep (principled_v1,
  p3_notag) sí que en fan servir: 1.40 i 1.90 m/s de velocitat mitjana dels jugadors per detectar pilota
  morta, heretats de markius_v1, d'origen no documentat.
- Distància cinemàtica invariant de contacte kappa <= 0.95 (adimensional) i decaïment 0.20 de la
  confiança física: constants heretades de starter/markius_v5.py, d'origen no documentat.
- Descodificació unificada a través del mateix motor Viterbi de models/decoder.py.
- Paràmetres generats estrictament amb bench/tune_decoder.py params.
- Nota sobre la pilota interpolada: principled_v1 interpola forats de pilota <= 25 frames.
  Per tant, phys_conf es calcula sobre la pilota interpolada i pot ser > 0 en forats curts;
  en forats llargs kappa es 1e9 i phys_conf = 0 (w_t = w).
- TCN neutra en forats (variant r): a r=0, p_tcn s'iguala al prior anul·lant el terme log(p/prior).
"""
import json
import os
import sys

ARREL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ARREL not in sys.path:
    sys.path.insert(0, ARREL)
BANC = os.path.join(ARREL, "bench")
if BANC not in sys.path:
    sys.path.insert(0, BANC)

import harness

PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "markius_v9_params.json")


def carrega_params(path=PARAMS):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No s'ha trobat el fitxer de paràmetres: {path}")
    with open(path) as fh:
        return json.load(fh)


def prediu(d, params=None):
    """Predicció de possessió al partit utilitzant els paràmetres ajustats."""
    params = params or carrega_params()
    # les regles de janv1 amb què es va calcular la graella (tune_decoder.py params --regles)
    pred_fn = harness._pred_decoder("fusio", "v7d", "v9", params.get("regles", "p3_notag"))
    return pred_fn(d, params=params)
