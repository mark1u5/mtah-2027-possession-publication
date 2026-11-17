"""Comprovació de muntatge del candidat v7. Només imprimeix agregats i no fa servir cap GT.

    $P bench/check_muntatge.py [--dominis partit idsse metrica]

Per a cada part de cada domini comprova dues coses:

- direcció: el signe de features.direccio coincideix amb el que surt de la posició mitjana dels
  porters (l'slot de cada equip amb la |x| mitjana més gran) al primer minut des del xut inicial;
- TCN contra regles: als frames clars de la porta, l'argmax de la TCN (--tcn, v7b per defecte) coincideix més amb
  l'etiqueta de les regles tal com és que intercanviant H i A.

Al partit es fa amb les regles p3 i notag; als externs (IDSSE i Metrica, llavors 0, 1 i 2), amb notag
i les probabilitats fora de plec. Surt amb 1 si alguna comprovació falla. make_submission.py
--model v7 fa la part del partit abans d'escriure el CSV.
"""
import argparse
import glob
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
ARREL = os.path.dirname(AQUI)
sys.path.insert(0, ARREL)
sys.path.insert(0, AQUI)
import adapter  # noqa: E402
import features as ft  # noqa: E402

FPS = adapter.FPS
FINESTRA_PORTER_S = 60
CLASSES = np.array(["H", "A", "D"])
INTERCANVIA = np.array([1, 0, 2])  # H <-> A


def signe_porters(d, xc, h, a, b):
    """+1 si el porter local és a x més petita que el visitant (el local ataca cap a +x)."""
    k0 = ft.xut_inicial(d, h, a, b)
    x = xc[k0:min(b + 1, k0 + FINESTRA_PORTER_S * FPS)]
    eq = d["team_of_track"]
    porter = []
    for k in (0, 1):
        cols = x[:, eq == k]
        pres = np.isfinite(cols).mean(0) if cols.size else np.zeros(0)
        with np.errstate(all="ignore"):
            mx = np.array([np.nanmean(cols[:, j]) if pres[j] >= 0.5 else np.nan for j in range(cols.shape[1])])
        porter.append(mx[np.nanargmax(np.abs(mx))] if np.isfinite(mx).any() else np.nan)
    if not np.isfinite(porter).all():
        return None
    return 1.0 if porter[0] < porter[1] else -1.0


def direccions(d):
    """{part: (signe de features, signe dels porters)}"""
    L, _ = d["pitch"]
    xc = d["px64"] - L / 2
    _, info = ft.direccio(d, xc)
    return {int(h): (info[int(h)]["signe"], signe_porters(d, xc, h, a, b)) for h, a, b in d["half_blocks"]}


def coincidencies(lab, clar, p, periode):
    """{part: (frames clars, coincidents, coincidents amb H i A intercanviats)}"""
    lab = np.asarray(lab).astype("<U1")
    am = p.argmax(1)
    out = {}
    for h in (1, 2):
        c = (periode == h) & np.asarray(clar, bool) & np.isin(lab, CLASSES)
        out[h] = (int(c.sum()), int((CLASSES[am[c]] == lab[c]).sum()), int((CLASSES[INTERCANVIA[am[c]]] == lab[c]).sum()))
    return out


def files(domini, coin, dirs):
    """Una fila per part: coin {part: [(n, ok, ok_x)]}, dirs {part: [(signe, signe porters)]}."""
    out = []
    for h in (1, 2):
        n = sum(c[0] for c in coin[h])
        norm = 100.0 * sum(c[1] for c in coin[h]) / max(n, 1)
        inter = 100.0 * sum(c[2] for c in coin[h]) / max(n, 1)
        ok_dir = sum(1 for s, g in dirs[h] if g is not None and s == g)
        out.append({"domini": domini, "part": h, "normal": norm, "intercanviada": inter,
                    "direccio_ok": ok_dir, "n": len(dirs[h]),
                    "passa": n > 0 and norm > inter and ok_dir == len(dirs[h])})
    return out


def comprova_partit(d, p, regles=None):
    """Files del partit. regles: {nom: (etiquetes, control clar)}; si falta, es calcula."""
    from models import v7
    regles = dict(regles or {})
    for cfg in ("p3", "p3_notag"):
        if cfg not in regles:
            regles[cfg] = v7.regles(d, cfg)
    per = ft.periode_de_joc(d)
    dirs = direccions(d)
    out = []
    for cfg, (lab, clar) in regles.items():
        c = coincidencies(lab, clar, p, per)
        out += files(f"partit {'notag' if cfg == 'p3_notag' else cfg}", {h: [c[h]] for h in (1, 2)},
                     {h: [dirs[h]] for h in (1, 2)})
    return out


def comprova_extern(font, variant="v7b"):
    import tune_decoder as td
    coin, dirs = {1: [], 2: []}, {1: [], 2: []}
    for f in sorted(glob.glob(os.path.join(td.DADES, font, "*.npz"))):
        partit, ll = os.path.basename(f)[:-4].rsplit("_s", 1)
        e = td.entrada(font, partit, int(ll), variant)
        d = adapter.carrega_extern(f)
        c = coincidencies(e["janv1"], e["clar"], e["p"], ft.periode_de_joc(d))
        dd = direccions(d)
        for h in (1, 2):
            coin[h].append(c[h])
            if h in dd:
                dirs[h].append(dd[h])
    return files(f"{font} notag", coin, dirs)


def imprimeix(rows):
    print("| domini | part | coincidència normal % | intercanviada % | direcció OK | passa |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['domini']} | {r['part']} | {r['normal']:.1f} | {r['intercanviada']:.1f} | "
              f"{r['direccio_ok']}/{r['n']} | {'sí' if r['passa'] else 'NO'} |", flush=True)
    return all(r["passa"] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dominis", nargs="+", default=["partit", "idsse", "metrica"])
    ap.add_argument("--tcn", default="v7b", help="variant de la TCN (v7b per a la v7, v7d per a la v7d)")
    a = ap.parse_args()
    rows = []
    if "partit" in a.dominis:
        from models import v7
        d = adapter.carrega_partit()
        rows += comprova_partit(d, v7.probs_tcn(d, v7.pesos(a.tcn)))
    for font in ("idsse", "metrica"):
        if font in a.dominis:
            rows += comprova_extern(font, a.tcn)
    ok = imprimeix(rows)
    print("muntatge correcte" if ok else "ERROR de muntatge")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
