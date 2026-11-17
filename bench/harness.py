"""Arnès d'avaluació externa: rànquing de models sobre el banc degradat.

    /workspace/shared/venvs/bench/bin/python bench/harness.py --models baseline_smooth50 janv1_notag \
        [--seeds 0] [--labels label_acc] [--suavitzat si|no] [--jobs 2]

Banc de selecció: els 10 partits d'IDSSE i Metrica de versio.DADES (degradació v2 per defecte). SkillCorner
s'informa en una columna a part i serveix per entrenar, però no decideix (docs/criteri_seleccio.md). Les mètriques són les de evaluation/possession_metrics.py. Del
partit proporcionat només se'n treuen agregats (sanitat) i el tram públic, on no s'hi decideix res.
Sortida: bench/results/<data>_<hora>_<models>.json i .md, només amb agregats.
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
ARREL = os.path.dirname(AQUI)
sys.path.insert(0, ARREL)
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(ARREL, "evaluation"))
import adapter  # noqa: E402
import possession_metrics as pm  # noqa: E402
import versio  # noqa: E402

DADES = versio.DADES
SELECCIO = ("idsse", "metrica")      # decideixen
INFORMATIU = ("skillcorner",)       # s'informa, no decideix
ENTRENAMENT = ("idsse", "skillcorner")  # per a models amb ajust (Metrica no s'hi fa servir)
FONTS = SELECCIO + INFORMATIU
TRAM_MIN = 60 * adapter.FPS   # trams vàlids de menys d'un minut no s'avaluen
N_BOOT = 2000
LLAVOR_BOOT = 0


# ---------------------------------------------------------------- models

def pred_baseline_smooth50(d, params=None):
    from starter import baseline
    codes = baseline.nearest_from_arrays(d["px64"], d["py64"], d["bx64"], d["by64"], d["team_of_track"])
    return baseline.smooth(codes, 50)


def _janv1(d, config):
    from starter import principled_v1 as pv
    cfg = pv.CONFIGS[config]
    kin = pv.kinematics_from_arrays(d["px"].copy(), d["py"].copy(), d["mean_speed"].copy(),
                                    d["bx"].copy(), d["by"].copy(), d["team_of_track"], d["pitch"])
    st = {"home": None, "away": None, "set_pieces": None}
    if cfg["visible_dead"] == "st_veto" or cfg["init"] == "v1":
        if d["domini"] != "partit":
            raise RuntimeError(f"{config} necessita Smart Tagging, que els externs no tenen")
        st["home"], st["away"], st["set_pieces"] = pv.load_smart_tagging_phases(d["smart_tagging"], d["n_frames"])
    labels, _ = pv.predict(kin, st, d["n_frames"], d["half_blocks"], d["kickoff_2_frame"], d["pitch"], cfg)
    return labels


def pred_janv1_notag(d, params=None):
    return _janv1(d, "p3_notag")


def pred_janv1_p3(d, params=None):
    return _janv1(d, "p3")


def fit_tcn(v):
    def fit(entrenament):
        """Checkpoint del plec que entrena amb aquests partits (bench/train_tcn.py cv)."""
        import train_tcn as tt
        claus = {_clau(f) for f in entrenament}
        for nom, av, ent, val in tt.plecs():
            if set(ent) | {val} == claus:
                path = tt.ruta_plec(nom, v)
                if not os.path.exists(path):
                    sys.exit(f"falta {path}: executa bench/train_tcn.py cv")
                return path
        sys.exit("cap plec de bench/train_tcn.py no entrena amb aquests partits")
    return fit


def pred_tcn(v):
    def pred(d, params=None):
        """Argmax per frame, sense descodificar. Als externs, el model del plec; al partit, el final."""
        import features as ft
        import train_tcn as tt
        from models import tcn_v7
        model, ck = tcn_v7.carrega(params or tt.rutes(v)["pesos"])
        if "_X" not in d:
            d["_X"] = ft.features(d)[0]
        p = tcn_v7.probabilitats(model, ck, d["_X"])
        return np.array(tcn_v7.CLASSES)[p.argmax(1)], p
    return pred


def _p_tcn(d, v):
    """Probabilitats de la TCN: als externs, les fora de plec desades; al partit, el model final."""
    k = "_p_tcn_" + v
    if k not in d:
        if d["domini"] == "partit":
            d[k] = pred_tcn(v)(d)[1]
        else:
            import train_tcn as tt
            z = np.load(os.path.join(tt.rutes(v)["probs"], d["font"], f"{d['partit']}_s{d['llavor']}.npz"), allow_pickle=False)
            d[k] = z["probs"]
    return d[k]


def _fit_decoder(config, v, g="c2b"):
    def fit(entrenament):
        import tune_decoder as td
        return td.ajusta(config, {_clau(f) for f in entrenament}, v, g)
    return fit


def _pred_decoder(config, v, g="c2b"):
    def pred(d, params=None):
        import tune_decoder as td
        from models import decoder as dc
        if params is None:  # partit: estadístiques dels 27 i paràmetres triats amb els 7 d'IDSSE
            params = td.ajusta(config, set(td.plecs()["tots"][1]), v, g)
        labels, clar = td.janv1_amb_control(d)
        p = _p_tcn(d, v) if params["cfg"]["w"] > 0 else np.full((len(labels), 3), 1.0 / 3)
        conf = td.conf_fisica(d) if "factor" in params["cfg"] else None
        r = (np.isfinite(d["bx"]) & np.isfinite(d["by"])) if params["cfg"].get("r") else None
        return dc.descodifica(p, labels, ~clar, np.asarray(d["period"]), params["st"], [params["cfg"]], conf=conf, r=r)[0]
    return pred


# nom -> (predicció, ajust o None, dominis). Un model amb ajust rep fit(fitxers d'entrenament) ->
# params i es valida deixant un partit fora cada vegada. La predicció pot tornar les etiquetes o
# (etiquetes, probabilitats H/A/D).
MODELS = {
    "baseline_smooth50": (pred_baseline_smooth50, None, ("extern", "partit")),
    "janv1_notag": (pred_janv1_notag, None, ("extern", "partit")),
    "janv1": (pred_janv1_p3, None, ("partit",)),  # p3 amb Smart Tagging: només informatiu al partit
}
# TCN (C2, C2b) i descodificador de C3 (docs/descodificador.md). janv1_vit no fa servir la TCN.
VARIANTS_TCN = ("v7", "v7b", "v7c", "v7d")
MODELS["janv1_vit"] = (_pred_decoder("janv1_vit", "v7b"), _fit_decoder("janv1_vit", "v7b"), ("extern", "partit"))
for _v in VARIANTS_TCN:
    MODELS[f"tcn_{_v}"] = (pred_tcn(_v), fit_tcn(_v), ("extern", "partit"))
    for _c in ("porta", "fusio", "tcn_vit"):
        MODELS[f"{_c}_{_v}"] = (_pred_decoder(_c, _v), _fit_decoder(_c, _v), ("extern", "partit"))
# Graella "vora" (v7): lam ∈ {0,1; 0,15; 0,25}, només porta i tcn_vit amb v7b.
for _c in ("porta", "tcn_vit"):
    MODELS[f"{_c}_v7b_vora"] = (_pred_decoder(_c, "v7b", "vora"), _fit_decoder(_c, "v7b", "vora"), ("extern", "partit"))
# C2c: la v7d amb la graella "c2c" (lam ∈ {0,05; 0,1; 0,15; 0,25}).
for _c in ("porta", "fusio", "tcn_vit"):
    MODELS[f"{_c}_v7d"] = (_pred_decoder(_c, "v7d", "c2c"), _fit_decoder(_c, "v7d", "c2c"), ("extern", "partit"))
# Markius v9: fusio_v7d amb modulació dinàmica per contacte cinemàtic (graella v9) i variant r (graella v9r)
MODELS["markius_v9"] = (_pred_decoder("fusio", "v7d", "v9"), _fit_decoder("fusio", "v7d", "v9"), ("extern", "partit"))
MODELS["markius_v9_r"] = (_pred_decoder("fusio", "v7d", "v9r"), _fit_decoder("fusio", "v7d", "v9r"), ("extern", "partit"))
MODELS["markius_v9_m1"] = (_pred_decoder("fusio", "v7d", "v9m1"), _fit_decoder("fusio", "v7d", "v9m1"), ("extern", "partit"))
# v9m1x: graella estesa de v9m1 (w ∈ {0,75; 1; 1,25}, factor ∈ {0,25; 0,35; 0,45; 0,6},
# lam ∈ {0,15; 0,25; 0,35; 0,5}, s_D ∈ {0,1; 0,25; 0,5; 1; 1,5}); 576 configs per plec
MODELS["markius_v9_m1x"] = (_pred_decoder("fusio", "v7d", "v9m1x"), _fit_decoder("fusio", "v7d", "v9m1x"), ("extern", "partit"))


def pred_v7(d, params=None):
    """Candidat v7 (models/v7.py): porta_v7b amb les regles p3. Només partit."""
    from models import v7
    return v7.prediu(d)[0]


MODELS["v7"] = (pred_v7, None, ("partit",))
# Models cars d'entrenar: a SkillCorner (informatiu) no es fa un plec per partit, es fa servir el
# plec de Metrica, que l'ha vist a l'entrenament. Aquests números són dins de mostra.
INFORMATIU_DINS = {m for m in MODELS if m.startswith(("tcn_", "porta_", "fusio_"))} | {"janv1_vit", "markius_v9", "markius_v9_r", "markius_v9_m1", "markius_v9_m1x"}
FORA = {"markius_v1": "fa servir el Smart Tagging", "markius_v6": "fa servir el Smart Tagging i una TCN entrenada amb Metrica"}


# ---------------------------------------------------------------- avaluació

def gt_avaluable(gt, clau):
    """Etiqueta de referència amb X als frames que no s'avaluen.

    No s'avaluen: valid = 0, fora de les parts, els trams vàlids de menys d'un minut, i el primer
    frame de cada part quan toca l'anterior (perquè cap possessió no travessi el descans)."""
    lab = np.asarray(gt[clau]).astype("<U1").copy()
    per = gt["period"]
    ok = (gt["valid"] == 1) & (per > 0) & (lab != "X")
    canvi = np.flatnonzero(per[1:] != per[:-1]) + 1
    ok[canvi] = False
    exclosos_curts = 0
    for a, b in adapter.ts.trams(ok):
        if b - a < TRAM_MIN:
            ok[a:b] = False
            exclosos_curts += b - a
    lab[~ok] = "X"
    vius = int(((gt["valid"] == 1) & (per > 0)).sum())
    return pm.label_codes(lab), {"frames_valids": vius, "frames_avaluats": int(ok.sum()),
                                 "exclosos_trams_curts": exclosos_curts}


N_BINS_ECE = 15


def ece(gt_codes, probs):
    """Error de calibratge esperat (15 intervals de la probabilitat màxima) als frames avaluats."""
    ok = gt_codes != pm.X
    conf = probs[ok].max(1)
    encert = (np.array([pm.H, pm.A, pm.D])[probs[ok].argmax(1)] == gt_codes[ok])
    bins = np.minimum((conf * N_BINS_ECE).astype(int), N_BINS_ECE - 1)
    e = 0.0
    for b in range(N_BINS_ECE):
        m = bins == b
        if m.any():
            e += m.mean() * abs(encert[m].mean() - conf[m].mean())
    return float(e)


def avalua(gt_codes, pred, probs=None):
    pc = pm.label_codes(pred)
    r = pm.evaluate(gt_codes, pc)
    ok = gt_codes != pm.X
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "precisio": float((pc[ok] == gt_codes[ok]).mean()),
            "ece": ece(gt_codes, probs) if probs is not None else None,
            "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]}


def feina_partit(args):
    path, models, etiquetes, smooth, params = args
    d = adapter.carrega_extern(path, smooth=smooth)
    gts = {e: gt_avaluable(d["gt"], e) for e in etiquetes}
    out = {"font": d["font"], "partit": d["partit"], "split": d["split"],
           "llavor": int(re.search(r"_s(\d+)\.npz$", path).group(1)), "cobertura": gts[etiquetes[0]][1], "models": {}}
    for m in models:
        pred = MODELS[m][0](d, params.get((m, d["font"], d["partit"])))
        probs = None
        if isinstance(pred, tuple):
            pred, probs = pred
        out["models"][m] = {e: avalua(gts[e][0], pred, probs) for e in etiquetes}
    return out


def _clau(f):
    return os.path.basename(os.path.dirname(f)), re.sub(r"_s\d+\.npz$", "", os.path.basename(f))


def ajusta_loo(models, fitxers, smooth):
    """Paràmetres per plec: per a cada partit, el model s'ajusta amb els partits d'entrenament que no
    són aquell (totes les llavors del mateix partit queden fora)."""
    params = {}
    pool = sorted(glob.glob(os.path.join(DADES, "*", "*.npz")))
    pool = [f for f in pool if _clau(f)[0] in ENTRENAMENT]
    for m in models:
        fit = MODELS[m][1]
        if fit is None:
            continue
        for k in sorted({_clau(f) for f in fitxers}):
            if k[0] in INFORMATIU and m in INFORMATIU_DINS:
                k_fora = ("metrica", None)
            else:
                k_fora = k
            params[(m,) + k] = fit([f for f in pool if _clau(f) != k_fora])
    return params


# ---------------------------------------------------------------- estadística entre partits

def bootstrap_mitjana(v, n=N_BOOT, llavor=LLAVOR_BOOT):
    v = np.asarray(v, float)
    rng = np.random.RandomState(llavor)
    mostres = v[rng.randint(0, len(v), (n, len(v)))].mean(axis=1)
    lo, hi = np.percentile(mostres, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def kendall_tau(a, b):
    """Tau-b de Kendall entre dues llistes de puntuacions."""
    n, c, dsc, ta, tb = len(a), 0, 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if x == 0 and y == 0:
                continue
            if x == 0:
                ta += 1
            elif y == 0:
                tb += 1
            elif x == y:
                c += 1
            else:
                dsc += 1
    den = np.sqrt((c + dsc + ta) * (c + dsc + tb))
    return float((c - dsc) / den) if den else None


# ---------------------------------------------------------------- partit proporcionat (agregats)

def sanitat_partit(labels, d, probs=None):
    codes = pm.label_codes(labels)
    runs = [(a, b + 1) for _, a, b in d["half_blocks"]]
    graded = np.zeros(len(codes), bool)
    for a, b in runs:
        graded[a:b] = True
    st = pm.possession_stats(pm.fill_abstentions(codes, runs), runs)
    g = codes[graded]
    out = {"possessions": st["n_spells"], "durada_mitjana_s": round(st["mean_duration_s"], 2),
           "pct_D": round(100.0 * float(np.mean(g == pm.D)), 2), "pct_X": round(100.0 * float(np.mean(g == pm.X)), 2)}
    if probs is not None:
        out["prob_mitjana_HAD"] = [round(float(x), 4) for x in probs[graded].mean(0)]
        out["pct_HAD"] = [round(100.0 * float(np.mean(g == c)), 2) for c in (pm.H, pm.A, pm.D)]
    return out


def partit(models, smooth):
    import domini
    from run_evaluation import find_gt
    pr = domini.prior()
    d = adapter.carrega_partit(smooth=smooth)
    gt = pm.label_codes(pm.read_label_csv(find_gt("public")))
    out = {}
    for m in models:
        labels, probs = MODELS[m][0](d), None
        if isinstance(labels, tuple):
            labels, probs = labels
        r = pm.evaluate(gt, pm.label_codes(labels))
        san = sanitat_partit(labels, d, probs)
        san["domini"] = domini.dins(san["possessions"], pr)
        out[m] = {"sanitat": san,
                  "public_no_s_hi_decideix": {"spellF1": round(r["spell_f1_mean"], 4), "macroF1": round(r["macro_f1"], 4)}}
    out["_prior_domini"] = pr
    return out


# ---------------------------------------------------------------- informe

METRIQUES = ("spellF1", "macroF1", "precisio", "ece")


def agrega(res, models, etiqueta, llavors):
    """Per model: mitjana per partit (sobre llavors) i agregats per font i sobre els de selecció."""
    per_partit = {}
    for r in res:
        if r["llavor"] not in llavors:
            continue
        k = (r["font"], r["partit"])
        for m in models:
            per_partit.setdefault((m, k), []).append(r["models"][m][etiqueta])
    taula = {}
    for m in models:
        files = {k: {x: (float(np.mean([v[x] for v in vals])) if all(v.get(x) is not None for v in vals) else None)
                     for x in METRIQUES} for (mm, k), vals in per_partit.items() if mm == m}
        sel = {k: v for k, v in files.items() if k[0] in SELECCIO}
        t = {"n_seleccio": len(sel), "per_partit": {f"{k[0]}/{k[1]}": v for k, v in sorted(files.items())}}
        for x in METRIQUES:
            vs = [v[x] for _, v in sorted(sel.items())]
            t[x] = bootstrap_mitjana(vs) if vs and None not in vs else None
            for f in FONTS:
                vv = [v[x] for k, v in files.items() if k[0] == f]
                t[f"{x}_{f}"] = float(np.mean(vv)) if vv and None not in vv else None
        t["_sel"] = {k: sel[k] for k in sorted(sel)}
        taula[m] = t
    return taula


def comparacions(taula, models):
    out = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            claus = sorted(set(taula[a]["_sel"]) & set(taula[b]["_sel"]))
            dif = [taula[b]["_sel"][k]["spellF1"] - taula[a]["_sel"][k]["spellF1"] for k in claus]
            mitj, lo, hi = bootstrap_mitjana(dif)
            out.append({"a": a, "b": b, "dif_spellF1_b_menys_a": mitj, "ic95": [lo, hi],
                        "b_guanya": lo > 0, "a_guanya": hi < 0, "partits": len(claus),
                        "b_millor_en_partits": int(sum(1 for x in dif if x > 0))})
    return out


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["baseline_smooth50", "janv1_notag"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--labels", default="label_acc")
    ap.add_argument("--data", default=DADES)
    ap.add_argument("--suavitzat", choices=["si", "no"], default=None,
                    help=f"Savitzky-Golay de les posicions (per defecte: {'si' if adapter.SUAVITZA else 'no'})")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--sense-partit", action="store_true", help="no calcula la sanitat del partit")
    a = ap.parse_args()
    a.smooth = adapter.SUAVITZA if a.suavitzat is None else a.suavitzat == "si"
    for m in a.models:
        if m in FORA:
            sys.exit(f"{m} queda fora de l'arnès: {FORA[m]} (vegeu docs/arnes.md)")
        if "extern" not in MODELS[m][2]:
            sys.exit(f"{m} només s'executa al partit")
    t0 = time.time()
    alt = "label_dur" if a.labels == "label_acc" else "label_acc"
    etiquetes = [a.labels, alt]
    fitxers = sorted(p for p in glob.glob(os.path.join(a.data, "*", "*.npz"))
                     if int(re.search(r"_s(\d+)\.npz$", p).group(1)) in a.seeds)
    params = ajusta_loo(a.models, fitxers, a.smooth)
    with Pool(a.jobs) as pool:
        res = pool.map(feina_partit, [(f, a.models, etiquetes, a.smooth, params) for f in fitxers])

    principal = agrega(res, a.models, a.labels, a.seeds)
    robust = agrega(res, a.models, alt, a.seeds)
    tau = kendall_tau([principal[m]["spellF1"][0] for m in a.models], [robust[m]["spellF1"][0] for m in a.models])
    per_llavor = {s: {m: agrega(res, a.models, a.labels, [s])[m]["spellF1"][0] for m in a.models} for s in a.seeds}
    rank_llavor = {s: sorted(a.models, key=lambda m: -v[m]) for s, v in per_llavor.items()}
    comp = comparacions(principal, a.models)
    cob = {}
    for f in FONTS:
        rr = [r["cobertura"] for r in res if r["font"] == f]
        if rr:
            cob[f] = {"pct_valids_avaluats": round(100.0 * sum(x["frames_avaluats"] for x in rr) / max(1, sum(x["frames_valids"] for x in rr)), 2),
                      "pct_valids_exclosos_trams_curts": round(100.0 * sum(x["exclosos_trams_curts"] for x in rr) / max(1, sum(x["frames_valids"] for x in rr)), 2)}
    mp = {} if a.sense_partit else partit(list(dict.fromkeys(a.models + ["janv1"])), a.smooth)
    temps = round(time.time() - t0, 1)

    ara = datetime.datetime.now()
    data = ara.strftime("%Y-%m-%d %H:%M")
    nom = (f"{ara:%Y-%m-%d_%H%M%S}_{'-'.join(a.models)}" + (f"_s{''.join(map(str, a.seeds))}" if a.seeds != [0] else "")
           + (f"_sg{adapter.FINESTRA}" if a.smooth else "_sensesuau"))
    os.makedirs(os.path.join(AQUI, "results"), exist_ok=True)
    try:
        import subprocess
        commit_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ARREL, text=True).strip()
    except Exception:
        commit_head = "desconegut"
    net = lambda t: {m: {k: v for k, v in x.items() if k != "_sel"} for m, x in t.items()}
    resultat = {"models": a.models, "llavors": a.seeds, "etiqueta": a.labels, "smooth": a.smooth,
                "git_head": commit_head,
                "principal": net(principal), "robustesa": {"etiqueta": alt, **net(robust)},
                "tau_kendall_acc_dur": tau, "spellF1_per_llavor": per_llavor, "ranquing_per_llavor": rank_llavor,
                "finestra_sg": adapter.FINESTRA if a.smooth else None,
                "finestra_derivades": adapter.FINESTRA if a.smooth else adapter.FINESTRA_VEL,
                "comparacions": comp, "cobertura": cob, "partit": mp, "temps_s": temps}
    with open(os.path.join(AQUI, "results", nom + ".json"), "w") as fh:
        json.dump(resultat, fh, indent=1, ensure_ascii=False, sort_keys=True)

    L = [f"# Arnès extern: {', '.join(a.models)}", "",
         f"Data {data}. Git HEAD `{commit_head}`. Llavors {a.seeds}. Etiqueta principal `{a.labels}`, robustesa `{alt}`. "
         f"Savitzky-Golay de les posicions: {'sí, finestra ' + str(adapter.FINESTRA) if a.smooth else 'no'}; derivades amb "
         f"finestra {adapter.FINESTRA if a.smooth else adapter.FINESTRA_VEL}. Temps total {temps} s.", "",
         "spellF1 i macroF1 de `evaluation/possession_metrics.py`. '10' = mitjana sobre els 10 partits de "
         "selecció (7 d'IDSSE i 3 de Metrica) amb l'interval del 95 % per bootstrap entre partits "
         f"({N_BOOT} remostrejos, llavor {LLAVOR_BOOT}). SkillCorner és informatiu. La columna pública és "
         "informativa: no s'hi decideix res.", "",
         "| model | spellF1 10 [IC95] | macroF1 10 [IC95] | IDSSE spellF1 / macroF1 | Metrica spellF1 / macroF1 "
         "| SkillCorner (informatiu) spellF1 / macroF1 | partit: poss. / durada s / % D / % X | domini | públic spellF1 / macroF1 (no s'hi decideix) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in sorted((set(a.models) | set(mp)) - {"_prior_domini"}, key=lambda m: -(principal[m]["spellF1"][0] if m in principal else -1)):
        p = principal.get(m)
        s = mp.get(m, {}).get("sanitat", {})
        q = mp.get(m, {}).get("public_no_s_hi_decideix", {})
        cel_p = (["{:.3f} [{:.3f}, {:.3f}]".format(*p["spellF1"]), "{:.3f} [{:.3f}, {:.3f}]".format(*p["macroF1"]),
                  f"{f3(p['spellF1_idsse'])} / {f3(p['macroF1_idsse'])}", f"{f3(p['spellF1_metrica'])} / {f3(p['macroF1_metrica'])}",
                  f"{f3(p['spellF1_skillcorner'])} / {f3(p['macroF1_skillcorner'])}"] if p else ["només partit (informatiu)", "-", "-", "-", "-"])
        cel_s = f"{s.get('possessions', '-')} / {s.get('durada_mitjana_s', '-')} / {s.get('pct_D', '-')} / {s.get('pct_X', '-')}" if s else "-"
        cel_q = f"{q.get('spellF1', '-')} / {q.get('macroF1', '-')}" if q else "-"
        cel_d = "-" if "domini" not in s else ("dins" if s["domini"] else "fora")
        L.append(f"| {m} | " + " | ".join(cel_p + [cel_s, cel_d, cel_q]) + " |")
    if "_prior_domini" in mp:
        pr = mp["_prior_domini"]
        L += ["", f"Comprovació de domini (docs/criteri_seleccio.md): prior {pr['prior']} possessions, "
              f"interval [{pr['interval'][0]}, {pr['interval'][1]}]. Un model fora de l'interval no es pot triar."]
    L += ["", "## Comparacions aparellades (spellF1, bootstrap per partits)", "",
          "| A | B | B − A | IC95 | partits on B > A | decisió |", "|---|---|---|---|---|---|"]
    for c in comp:
        dec = f"entra {c['b']}" if c["b_guanya"] else (f"entra {c['a']}" if c["a_guanya"] else "no es distingeixen: guanya el més simple")
        L.append(f"| {c['a']} | {c['b']} | {c['dif_spellF1_b_menys_a']:+.4f} | [{c['ic95'][0]:+.4f}, {c['ic95'][1]:+.4f}] "
                 f"| {c['b_millor_en_partits']}/{c['partits']} | {dec} |")
    L += ["", "## Robustesa", "",
          f"Tau de Kendall entre el rànquing amb `{a.labels}` i amb `{alt}` (spellF1 sobre els 10): {tau}.", "",
          "| model | spellF1 10 amb " + alt + " [IC95] | macroF1 10 amb " + alt + " |", "|---|---|---|"]
    for m in a.models:
        L.append(f"| {m} | " + "{:.3f} [{:.3f}, {:.3f}]".format(*robust[m]["spellF1"]) + f" | {robust[m]['macroF1'][0]:.3f} |")
    L += ["", "Rànquing per llavor (spellF1 sobre els 10):", ""]
    for s in a.seeds:
        L.append(f"- llavor {s}: " + " > ".join(f"{m} ({per_llavor[s][m]:.3f})" for m in rank_llavor[s]))
    L += ["", "## Per partit", "", "Mitjana sobre les llavors. spellF1 / macroF1 / precisió per frame / ECE "
          "(l'ECE només per als models que donen probabilitats)."
          + (" SkillCorner, per a " + ", ".join(sorted(INFORMATIU_DINS & set(a.models))) + ", és dins de l'entrenament."
             if INFORMATIU_DINS & set(a.models) else ""), "",
          "| partit | " + " | ".join(a.models) + " |", "|---|" + "---|" * len(a.models)]
    for k in sorted(principal[a.models[0]]["per_partit"], key=lambda k: (k.split("/")[0] not in SELECCIO, k)):
        cel = []
        for m in a.models:
            v = principal[m]["per_partit"].get(k, {})
            cel.append(" / ".join(f3(v.get(x)) for x in METRIQUES))
        L.append(f"| {k} | " + " | ".join(cel) + " |")
    L.append("| **mitjana dels 10** | " + " | ".join(
        " / ".join(f3(principal[m][x][0] if principal[m][x] else None) for x in METRIQUES) for m in a.models) + " |")
    for m in a.models:
        q = mp.get(m, {}).get("sanitat", {})
        if "prob_mitjana_HAD" in q:
            L += ["", f"Partit, {m}: probabilitat mitjana H/A/D {q['prob_mitjana_HAD']}, % d'argmax H/A/D {q['pct_HAD']}."]
    L += ["", "## Frames avaluats", "", "| font | % dels frames vàlids avaluats | % exclosos per trams < 1 min |", "|---|---|---|"]
    L += [f"| {f} | {c['pct_valids_avaluats']} | {c['pct_valids_exclosos_trams_curts']} |" for f, c in cob.items()]
    with open(os.path.join(AQUI, "results", nom + ".md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
