"""Comprova els partits degradats contra bench/target_stats.json i desa bench/report_degradacio.md.

Tolerància (bloqueja): pilota absent ±5 punts; mediana de durada dels forats ±30 %.
La resta (fragments, visibles, soroll, taxes) s'informa. Només agregats.

    /workspace/shared/venvs/bench/bin/python bench/check_degradation.py --dir /workspace/shared/bench_v7_deg [--check DIR2]
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import tracking_stats as ts  # noqa: E402

TOL_ABSENT = 5.0
TOL_MEDIANA = 0.30
FONTS = ["idsse", "skillcorner", "metrica"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloc in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def stats_fitxer(path):
    z = np.load(path, allow_pickle=False)
    camp = tuple(json.loads(str(z["meta"]))["camp_m"])
    s = ts.estadistiques(z["ball_xyz"][:, :2], z["players_xy"], z["players_team"], z["period"] > 0, camp=camp)
    s["forats_D"] = forats_D(z, z["ball_xyz"])
    return s


def forats_D(z, bola):
    """Frames de forat (pilota absent) que cauen en D, als frames vàlids amb etiqueta H/A/D."""
    lab = z["label_acc"].astype("<U1")
    ok = (z["period"] > 0) & (z["valid"] == 1) & np.isin(lab, ("H", "A", "D"))
    forat = ok & ~np.isfinite(bola[:, 0])
    return {"frames_forat": int(forat.sum()), "frames_forat_D": int((forat & (lab == "D")).sum()),
            "frames": int(ok.sum()), "frames_D": int((ok & (lab == "D")).sum())}


def agrega(llista):
    """Agrega partits: % absent ponderat per frames; quantils dels forats i fragments ponderant cada partit pel seu nombre."""
    fr = np.array([s["frames"] for s in llista], float)
    absent = float(np.sum([s["pilota"]["pct_absent"] * f for s, f in zip(llista, fr)]) / fr.sum())

    def pool(get):
        mostres = []
        for s in llista:
            d = get(s)
            if d.get("n", 0):
                # reconstrucció aproximada de la mostra a partir de la funció de quantils
                q = np.asarray(d["quantils101_frames"])
                mostres.append(np.interp(np.linspace(0, 100, d["n"]), np.arange(101), q))
        return np.concatenate(mostres) if mostres else np.zeros(0)

    forats = pool(lambda s: s["pilota"]["forats"])
    frags = pool(lambda s: s["jugadors"]["fragments"])
    taxes = {}
    for k in ("v<5", "v5-15", "v>15"):
        o = sum(s["pilota"]["inici_condicionat"].get(k, {}).get("oportunitats", 0) for s in llista)
        i = sum(s["pilota"]["inici_condicionat"].get(k, {}).get("inicis", 0) for s in llista)
        taxes[k] = round(i / o, 6) if o else None
    return {
        "pct_absent": round(absent, 2),
        "forats_q_s": {f"p{q}": round(float(np.percentile(forats, q)) / ts.FPS, 2) for q in ts.QUANTILS} if forats.size else {},
        "fragment_mediana_s": round(float(np.median(frags)) / ts.FPS, 2) if frags.size else None,
        "visibles_per_equip": round(float(np.mean([s["jugadors"]["visibles_mitjana"] for s in llista])), 2),
        "soroll_jugadors_m": round(float(np.mean([s["soroll"]["jugadors_std_m"] or 0 for s in llista])), 5),
        "soroll_pilota_m": round(float(np.mean([s["soroll"]["pilota_std_m"] or 0 for s in llista])), 5),
        "taxes_inici": taxes,
        "slots_mitjana": round(float(np.mean([s["jugadors"]["n_slots"] for s in llista])), 1),
        "taxes_v2": marginals([s["pilota"]["inici_v2"] for s in llista]),
        "pct_forat_D": pct_forat_D(llista),
    }


def marginals(v2s):
    """Taxa d'inici per franja (velocitat de la pilota, línia, jugadors), sumant partits."""
    cel = {}
    for v in v2s:
        for k, c in v["cel"].items():
            x = cel.setdefault(k, {"inicis": 0, "oportunitats": 0})
            x["inicis"] += c["inicis"]
            x["oportunitats"] += c["oportunitats"]
    return {k: v["taxa"] for k, v in ts.marginals_v2({"cel": cel}).items()}


def pct_forat_D(llista):
    if "forats_D" not in llista[0]:
        return None
    f = sum(s["forats_D"]["frames_forat"] for s in llista)
    fd = sum(s["forats_D"]["frames_forat_D"] for s in llista)
    n = sum(s["forats_D"]["frames"] for s in llista)
    nd = sum(s["forats_D"]["frames_D"] for s in llista)
    return {"pct_forat_en_D": round(100.0 * fd / f, 2) if f else None, "pct_D_tots": round(100.0 * nd / n, 2) if n else None}


def objectiu(t):
    p = t["pilota"]
    return {"pct_absent": p["pct_absent"],
            "forats_q_s": {k: round(v, 2) for k, v in p["forats"]["quantils_s"].items()},
            "fragment_mediana_s": t["jugadors"]["fragments"]["quantils_s"]["p50"],
            "visibles_per_equip": t["jugadors"]["visibles_mitjana"],
            "soroll_jugadors_m": t["soroll"]["jugadors_std_m"], "soroll_pilota_m": t["soroll"]["pilota_std_m"],
            "taxes_inici": {k: p["inici_condicionat"].get(k, {}).get("taxa_inici") for k in ("v<5", "v5-15", "v>15")},
            "slots_mitjana": t["jugadors"]["n_slots"],
            "taxes_v2": {k: v["taxa"] for k, v in ts.marginals_v2(p["inici_v2"]).items()} if "inici_v2" in p else {},
            "pct_forat_D": None}


def veredicte(a, o):
    pa = abs(a["pct_absent"] - o["pct_absent"]) <= TOL_ABSENT
    m, mo = a["forats_q_s"].get("p50"), o["forats_q_s"]["p50"]
    pm = m is not None and abs(m / mo - 1) <= TOL_MEDIANA
    return pa, pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/shared/bench_v7_deg")
    ap.add_argument("--orig", default="/workspace/shared/bench_v7", help="dades sense degradar, per referència")
    ap.add_argument("--check", default=None)
    ap.add_argument("--target", default=os.path.join(AQUI, "target_stats.json"))
    ap.add_argument("--nota-model", default=None, help="text sobre el canvi de model, si n'hi ha")
    ap.add_argument("--out", default=os.path.join(AQUI, "report_degradacio.md"))
    ap.add_argument("--params-v3", default=None, help="bench/forats_v3.json: afegeix l'acceptació del % de forat en D")
    a = ap.parse_args()
    with open(a.target) as fh:
        t = json.load(fh)
    obj = objectiu(t)

    files, grups, hashes = {}, {}, {}
    for path in sorted(glob.glob(os.path.join(a.dir, "*", "*.npz"))):
        font = os.path.basename(os.path.dirname(path))
        seed = int(re.search(r"_s(\d+)\.npz$", path).group(1))
        grups.setdefault((font, seed), []).append(stats_fitxer(path))
        hashes[os.path.relpath(path, a.dir)] = sha256(path)
    agregats = {k: agrega(v) for k, v in grups.items()}

    # referència: dades originals només amb deteccions reals (abans de degradar)
    ref = {}
    for font in FONTS:
        ll = []
        for path in sorted(glob.glob(os.path.join(a.orig, font, "*.npz"))):
            z = np.load(path, allow_pickle=False)
            b = z["ball_xyz"][:, :2].copy()
            b[~z["ball_detected"]] = np.nan
            xy = z["players_xy"].copy()
            xy[~z["players_detected"]] = np.nan
            camp = tuple(json.loads(str(z["meta"]))["camp_m"])
            st = ts.estadistiques(b, xy, z["players_team"], z["period"] > 0, camp=camp)
            st["forats_D"] = forats_D(z, b)
            ll.append(st)
        if ll:
            ref[font] = agrega(ll)

    det = None
    if a.check:
        iguals = sum(1 for k, h in hashes.items()
                     if os.path.exists(os.path.join(a.check, k)) and sha256(os.path.join(a.check, k)) == h)
        det = {"iguals": iguals, "total": len(hashes)}

    def fila(nom, x, verd=None):
        q = x["forats_q_s"]
        cel = [nom, f"{x['pct_absent']:g}", " / ".join(f"{q.get(k, '-')}" for k in ("p10", "p25", "p50", "p75", "p90")),
               " / ".join(f"{x['taxes_inici'][k]}" for k in ("v<5", "v5-15", "v>15")),
               f"{x['fragment_mediana_s']}", f"{x['visibles_per_equip']}", f"{x['soroll_jugadors_m']}",
               f"{x['soroll_pilota_m']}", f"{x['slots_mitjana']}"]
        if verd is not None:
            cel += ["passa" if verd[0] else "no passa", "passa" if verd[1] else "no passa"]
        else:
            cel += ["", ""]
        return "| " + " | ".join(cel) + " |"

    cap = ("| fila | % pilota absent | forats p10/p25/p50/p75/p90 s | taxa inici v<5 / 5-15 / >15 m/s "
           "| fragment p50 s | visibles/equip | soroll jug. m | soroll pilota m | slots | absent ±5 | mediana ±30 % |")
    out = ["# Degradació dels partits externs", "",
           "Generat per `bench/check_degradation.py`. Objectiu: `bench/target_stats.json` (agregats del "
           "tracking automàtic del partit proporcionat, sense GT). Frames: les parts (period > 0). "
           f"Tolerància que bloqueja: pilota absent ±{TOL_ABSENT:g} punts i mediana dels forats ±{TOL_MEDIANA:.0%}. "
           "La resta s'informa. 'original' = dades sense degradar, comptant només deteccions reals.", ""]
    if a.nota_model:
        out += ["## Canvi de model", "", a.nota_model, ""]
    if det:
        out += ["## Determinisme", "", f"Dues execucions de `degrade.py`: {det['iguals']} de {det['total']} "
                "fitxers amb el mateix sha256.", ""]
    tot_passa = True
    for font in FONTS:
        seeds = sorted(s for f, s in agregats if f == font)
        if not seeds:
            continue
        out += [f"## {font}", "", cap, "|---" * 11 + "|", fila("objectiu", obj)]
        if font in ref:
            out.append(fila("original", ref[font]))
        for s in seeds:
            v = veredicte(agregats[(font, s)], obj)
            tot_passa &= all(v)
            out.append(fila(f"llavor {s}", agregats[(font, s)], v))
        out.append("")
        franges = list(ts.VEL_CLAUS[:3]) + list(ts.LINIA_FRANGES) + list(ts.JUG_FRANGES)
        out += [f"Taxa d'inici de forat per franja ({font}). Línia: distància de la pilota a la línia més propera, "
                "en m. Jugadors: velocitat mitjana dels visibles, en m/s. Última columna: % dels frames de forat "
                "que cauen en D (i % de D a tots els frames), amb `label_acc`.", "",
                "| fila | " + " | ".join(franges) + " | forat en D % (D a tots %) |", "|---" * (len(franges) + 2) + "|"]
        files_v2 = [("objectiu", obj)] + ([("original", ref[font])] if font in ref else []) + \
                   [(f"llavor {s}", agregats[(font, s)]) for s in seeds]
        for nom, x in files_v2:
            fd = x.get("pct_forat_D")
            cel = [nom] + [f"{x['taxes_v2'].get(k):.5f}" if x["taxes_v2"].get(k) is not None else "-" for k in franges]
            cel.append(f"{fd['pct_forat_en_D']} ({fd['pct_D_tots']})" if fd else "-")
            out.append("| " + " | ".join(cel) + " |")
        out.append("")
    acc_D = None
    if a.params_v3:
        with open(a.params_v3) as fh:
            p3 = json.load(fh)
        ref_D = agregats[("skillcorner", 0)]["pct_forat_D"]["pct_forat_en_D"]
        net_D = ref["skillcorner"]["pct_forat_D"]["pct_forat_en_D"]
        r = p3["resum"]
        out += ["## Degradació v3: forats en D", "",
                f"k_viva = {p3['k_viva']:.4f} i k_D = {p3['k_D']:.4f}, fixats amb la llavor {p3['calibratge']['llavor']} "
                f"(bench/forats_v3.py). Referència del % de frames de forat en D: SkillCorner amb el mateix ompliment, "
                f"{ref_D} % (sense degradar, {net_D} %). Tolerància: ±{TOL_ABSENT:g} punts. Passa per construcció a la "
                "llavor de calibratge.", "",
                "| IDSSE | % absent | mediana dels forats s | % forat en D | absent ±5 | mediana ±30 % | forat en D ±5 |",
                "|---|---|---|---|---|---|---|"]
        acc_D = {}
        for s_ in sorted(s for f, s in agregats if f == "idsse"):
            x = agregats[("idsse", s_)]
            v = veredicte(x, obj)
            pd = x["pct_forat_D"]["pct_forat_en_D"]
            okd = abs(pd - ref_D) <= TOL_ABSENT
            acc_D[s_] = {"pct_absent": x["pct_absent"], "mediana_s": x["forats_q_s"].get("p50"), "pct_forat_D": pd,
                         "absent": v[0], "mediana": v[1], "forat_D": okd}
            out.append(f"| llavor {s_} | {x['pct_absent']} | {x['forats_q_s'].get('p50')} | {pd} | "
                       + " | ".join("passa" if q else "no passa" for q in (v[0], v[1], okd)) + " |")
        out += ["", "Quocients r = taxa(D) / taxa(viva) a SkillCorner sense degradar (frames vàlids, `label_acc`). "
                "S'informen; la v3 no els fa servir directament.", "",
                f"Global: {r['r_global']:.2f} (taxa viva {r['taxa_viva']:.5f}, D {r['taxa_D']:.5f}). Per cel·la agrupada: "
                f"mínim {r['r_cel·les']['min']:.2f}, mediana {r['r_cel·les']['mediana']:.2f}, màxim {r['r_cel·les']['max']:.2f}. "
                f"Durada mediana dels forats: {r['durada_mediana_s']['viva']} s amb la pilota viva i {r['durada_mediana_s']['D']} s amb la morta.", "",
                "| franja | r | inicis viva / D |", "|---|---|---|"]
        for fr_, m in r["marginals"].items():
            out.append(f"| {fr_} | {'-' if m['r'] is None else round(m['r'], 2)} | {m['inicis_viva']} / {m['inicis_D']} |")
        out.append("")
    out += ["## sha256", "", "| fitxer | sha256 |", "|---|---|"] + [f"| {k} | `{h}` |" for k, h in sorted(hashes.items())]
    with open(a.out, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(json.dumps({"objectiu": obj, "original": ref,
                      "degradat": {f"{f}/s{s}": {**v, "passa": veredicte(v, obj)} for (f, s), v in sorted(agregats.items())},
                      "determinisme": det, "tot_passa": tot_passa, "acceptacio_v3": acc_D}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
