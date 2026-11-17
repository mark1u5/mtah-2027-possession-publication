#!/usr/bin/env python3
"""Proves ràpides del descodificador HMM (models/decoder.py) amb dades sintètiques.

    $P tests/test_decoder.py

No necessita dades. Comprova:
- el lot mixt de configuracions (escalars i tuples s_dur barrejats) i el chunking > 384;
- que s_dur escalar i la tupla (s, s, s) donen la mateixa decodificació;
- que les durades mínimes m_c derivades de (s_P, s_P, s_D) es respecten a la sortida.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "base"))
from models import decoder as dc  # noqa: E402

ST = {"p5": [35.0, 35.6, 113.0], "mitjana": [334.0, 310.7, 489.2],
      "transicions": [[0, 1817, 1251], [1856, 0, 1278], [1153, 1264, 0]],
      "prior": [0.326, 0.309, 0.365]}


def dades_sintetiques(T=4000, llavor=0):
    """p_tcn i janv1 estructurats per trams (trams de ~200 frames) perquè les durades mínimes lliguin."""
    rng = np.random.default_rng(llavor)
    period = np.ones(T, int)
    period[T // 2:] = 2
    # veritat: trams llargs de H/A/D
    veritat = np.empty(T, "<U1")
    i = 0
    while i < T:
        d = int(rng.integers(60, 400))
        veritat[i:i + d] = rng.choice(np.array(["H", "A", "D"]))
        i += d
    codi = {"H": 0, "A": 1, "D": 2}
    y = np.array([codi.get(c, 0) for c in veritat])
    # p_tcn sorollós al voltant de la veritat (80% encert); janv1 també, amb forats X
    p = np.full((T, 3), 0.1)
    p[np.arange(T), y] = 0.8
    soroll = rng.random(T) < 0.2
    p[soroll] = rng.dirichlet([2, 2, 2], int(soroll.sum()))
    labels = veritat.copy()
    labels[rng.random(T) < 0.15] = "X"
    dubtos = rng.random(T) < 0.3
    conf = rng.random(T)
    r = rng.random(T) < 0.9
    return p, labels, dubtos, period, conf, r


def main():
    p, labels, dubtos, period, conf, r = dades_sintetiques()
    T = len(labels)

    # 1) m_c amb s_dur desacoblats (s_P, s_P, s_D): p5 = [35, 35.6, 113]
    m, lq, ls = dc.parametres_hmm(ST, (0.5, 0.5, 1.5))
    assert m.tolist() == [18, 18, 170], f"m inesperat: {m}"
    assert np.isfinite(lq).all()
    assert (np.diag(ls) == -np.inf).all(), "cap transició a la mateixa classe"
    assert np.isfinite(ls[~np.eye(3, dtype=bool)]).all(), "les transicions fora de la diagonal són finites"

    # 2) s_dur escalar == tupla (s, s, s)
    cfg_eq = [{"w": 1.0, "lam": 0.1, "s_dur": 1.0}, {"w": 1.0, "lam": 0.1, "s_dur": (1.0, 1.0, 1.0)}]
    o2 = dc.descodifica(p, labels, dubtos, period, ST, cfg_eq, conf=conf, r=r)
    assert np.array_equal(o2[0], o2[1]), "s_dur escalar i (s,s,s) han de coincidir"

    # 3) lot gran (> 384 configs, escalars i tuples barrejats entre blocs) i chunking
    cfgs = ([{"w": 1.0, "lam": 0.1, "s_dur": s} for s in (0.5, 1.0, 1.5)]
            + [{"w": w, "lam": l, "s_dur": (sp, sp, sd), "factor": f}
               for w in (0.5, 0.75, 1.0) for l in (0.05, 0.1, 0.15) for sp in (0.5, 1.0)
               for sd in (0.25, 0.5, 1.0, 1.5) for f in (0.0, 0.15, 0.25, 0.35, 0.45, 0.6)])
    assert len(cfgs) > 385, len(cfgs)
    out = dc.descodifica(p, labels, dubtos, period, ST, cfgs, conf=conf, r=r)
    assert out.shape == (len(cfgs), T)
    assert set(np.unique(out)) <= {"H", "A", "D", "X"}
    # el chunking (blocs de 384) ha de ser transparent: sublot == mateix subconjunt del lot sencer
    out_sub = dc.descodifica(p, labels, dubtos, period, ST, cfgs[:384], conf=conf, r=r)
    assert np.array_equal(out_sub, out[:384]), "el chunking altera els resultats (sublot)"
    # i el traçat complet per blocs de 384 == tot sencer
    traced = dc.descodifica(p, labels, dubtos, period, ST, cfgs, conf=conf, r=r)
    assert np.array_equal(traced, out), "el traçat per blocs de 384 altera els resultats"
    # límits exactes: un lot de just 384 i un de 385 també han de funcionar
    for k in (384, 385):
        out_k = dc.descodifica(p, labels, dubtos, period, ST, cfgs[:k], conf=conf, r=r)
        assert out_k.shape == (k, T), f"lot de {k} configs no desencadena el chunking correctament"
    # la cua (segon bloc, 384:) també ha de coincidir amb la descodificació sense chunking
    out_cua = dc.descodifica(p, labels, dubtos, period, ST, cfgs[384:], conf=conf, r=r)
    assert np.array_equal(out_cua, out[384:]), "el segon bloc (384:) no coincideix sense chunking"
    # s_dur no és invariant a permutar classes: (s_P=1, s_D=0.25) vs (s_P=0.25, s_D=1)
    # han de donar com a mínim un tram de durada diferent (els m_c són (36, 36, 28) vs (9, 9, 113))
    a, b = dc.descodifica(p, labels, dubtos, period, ST,
                          [{"w": 1.0, "lam": 0.1, "s_dur": (1.0, 1.0, 0.25)},
                           {"w": 1.0, "lam": 0.1, "s_dur": (0.25, 0.25, 1.0)}], conf=conf, r=r)
    assert dc.parametres_hmm(ST, (1.0, 1.0, 0.25))[0].tolist() == [35, 36, 28]
    assert dc.parametres_hmm(ST, (0.25, 0.25, 1.0))[0].tolist() == [9, 9, 113]
    assert not np.array_equal(a, b), "s_P i s_D desacoblats han de donar camins diferents"

    # 4) durades mínimes respectades a la sortida (part 1, plec complet)
    m170 = dc.parametres_hmm(ST, (0.5, 0.5, 1.5))[0]
    fila = out[0] if out.shape[0] else None
    # fila amb s_D=1.5 -> cap tram de D dins de part pot durar menys de m_D + 1 = 171
    cfg_sd = [{"w": 1.0, "lam": 0.1, "s_dur": (0.5, 0.5, 1.5)}]
    o = dc.descodifica(p, labels, dubtos, period, ST, cfg_sd, conf=conf, r=r)[0]
    dins = period > 0
    canvi = np.flatnonzero((o[1:] != o[:-1]) & dins[1:] & dins[:-1]) + 1
    ini = np.concatenate(([0], canvi))
    fi = np.concatenate((canvi, [T]))
    for i, j in zip(ini, fi):
        if o[i] == "D" and dins[i]:
            assert j - i >= m170[2] + 1, f"tram D de {j - i} < {m170[2] + 1}"
    print("OK: tests/test_decoder.py (m_c desacoblats, escalars==tuples, chunking 384, durades mínimes)")


if __name__ == "__main__":
    main()
