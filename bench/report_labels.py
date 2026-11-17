"""Informe agregat de les etiquetes externes generades per build_labels.py.

Només imprimeix i desa agregats (bench/report_etiquetes.md). Amb --check compara els sha256
amb una segona generació per comprovar el determinisme.

    /workspace/shared/venvs/bench/bin/python bench/report_labels.py --dir /workspace/shared/bench_v7 [--check DIR2]
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
FONTS = ["idsse", "skillcorner", "metrica"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloc in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def trams(v):
    if v.size == 0:
        return []
    talls = np.flatnonzero(v[1:] != v[:-1]) + 1
    ini = np.concatenate([[0], talls])
    fi = np.concatenate([talls, [v.size]])
    return [(v[i], int(f - i)) for i, f in zip(ini, fi)]


def stats(lab, valid, periode, fps):
    """Possessions = trams màxims de H o A (com al concurs), dins de cada part i de frames vàlids."""
    v = valid == 1
    nv = int(v.sum())
    r = {c: round(100.0 * float((lab[v] == c).sum()) / nv, 2) if nv else None for c in "HAD"}
    poss, morta, directes = [], [], 0
    # els frames no vàlids tallen els trams
    clau = np.where(v, lab, "#")
    for p in sorted(set(periode[periode > 0].tolist())):
        tr = trams(clau[periode == p])
        for i, (c, ll) in enumerate(tr):
            if c in ("H", "A"):
                poss.append(ll)
                if i + 1 < len(tr) and tr[i + 1][0] in ("H", "A"):
                    directes += 1
            elif c == "D":
                morta.append(ll)
    a, m = np.array(poss, float) / fps, np.array(morta, float) / fps
    r.update({"possessions": len(poss), "poss_mitjana_s": round(float(a.mean()), 2) if a.size else None,
              "poss_mediana_s": round(float(np.median(a)), 2) if a.size else None,
              "morta_trams": len(morta), "morta_mitjana_s": round(float(m.mean()), 2) if m.size else None,
              "morta_mediana_s": round(float(np.median(m)), 2) if m.size else None, "canvis_directes": directes})
    return r


def partit(path):
    z = np.load(path, allow_pickle=False)
    m = json.loads(str(z["meta"]))
    fps = float(m["fps"])
    valid, per = z["valid"], z["period"]
    r = {"font": m["font"], "partit": m["partit"], "split": m["split"], "frames": int(valid.size),
         "pct_no_valids": round(100.0 * float((valid == 0).mean()), 2),
         "acc": stats(z["label_acc"], valid, per, fps), "dur": stats(z["label_dur"], valid, per, fps),
         "prov": stats(z["label_prov"], valid, per, fps) if m["font"] != "metrica" else None,
         "cru": stats(z["label_cru"], valid, per, fps),
         "comprovacio": m.get("comprovacio"), "sha256": sha256(path)}
    v = valid == 1
    r["acc_dur_pct_iguals"] = round(100.0 * float((z["label_acc"][v] == z["label_dur"][v]).mean()), 2)
    r["acc_dur_dif_possessions"] = r["dur"]["possessions"] - r["acc"]["possessions"]
    if m["font"] == "skillcorner":
        L, W = m["camp_m"]
        b = z["ball_xyz"]
        d = v & (z["label_acc"] == "D") & np.isfinite(b[:, 0])
        dins = (np.abs(b[:, 0]) <= L / 2) & (np.abs(b[:, 1]) <= W / 2)
        r["pct_D_pilota_dins"] = round(100.0 * float((d & dins).sum()) / d.sum(), 2) if d.sum() else None
    for k in ("n_parades_amb_sortida_detectada", "n_represes_sense_obertura", "n_tancades_sense_marca_represa",
              "n_set_pieces_sense_disparador", "n_accions_controlades", "sincronitzacio", "camp_suposat"):
        if k in m:
            r[k] = m[k]
    return r


def f(x):
    return "-" if x is None else (f"{x:g}" if isinstance(x, (int, float)) else str(x))


def taula_partits(recs):
    out = ["| partit | % H | % A | % D | % no vàlids | poss. | poss. mitj./med. s | D mitj./med. s | H↔A directes "
           "| poss. (prov.) | poss. prov. mitj./med. s | H↔A dir. (prov.) | poss. (cru) | acc=dur % "
           "| Δposs. dur-acc | comprovació % |",
           "|---" * 16 + "|"]
    for r in recs:
        a, p = r["acc"], r["prov"] or {}
        c = r["comprovacio"] or {}
        out.append("| " + " | ".join([
            r["partit"], f(a["H"]), f(a["A"]), f(a["D"]), f(r["pct_no_valids"]), f(a["possessions"]),
            f"{f(a['poss_mitjana_s'])} / {f(a['poss_mediana_s'])}",
            f"{f(a['morta_mitjana_s'])} / {f(a['morta_mediana_s'])}", f(a["canvis_directes"]),
            f(p.get("possessions")), f"{f(p.get('poss_mitjana_s'))} / {f(p.get('poss_mediana_s'))}",
            f(p.get("canvis_directes")), f(r["cru"]["possessions"]),
            f(r["acc_dur_pct_iguals"]), f(r["acc_dur_dif_possessions"]), f(c.get("pct"))]) + " |")
    return out


def mitj(recs, f_):
    v = [x for x in (f_(r) for r in recs) if x is not None]
    return round(float(np.mean(v)), 2) if v else None


def resum_font(recs):
    s = {"partits": len(recs)}
    for var in ("acc", "prov", "cru", "dur"):
        if recs[0][var] is None:
            continue
        for k in ("possessions", "poss_mitjana_s", "poss_mediana_s", "D", "morta_mitjana_s", "morta_mediana_s",
                  "canvis_directes"):
            s[f"{var}.{k}"] = mitj(recs, lambda r: r[var][k])
    s["pct_no_valids"] = mitj(recs, lambda r: r["pct_no_valids"])
    s["acc_dur_pct_iguals"] = mitj(recs, lambda r: r["acc_dur_pct_iguals"])
    s["acc_dur_dif_possessions"] = mitj(recs, lambda r: r["acc_dur_dif_possessions"])
    c = [r["comprovacio"]["pct"] for r in recs if r.get("comprovacio")]
    if c:
        s["comprovacio_min_pct"] = min(c)
        s["comprovacio_mitj_pct"] = round(float(np.mean(c)), 2)
    if "pct_D_pilota_dins" in recs[0]:
        s["pct_D_pilota_dins"] = mitj(recs, lambda r: r["pct_D_pilota_dins"])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/shared/bench_v7")
    ap.add_argument("--check", default=None, help="segona generació per comparar sha256")
    ap.add_argument("--out", default=os.path.join(AQUI, "report_etiquetes.md"))
    a = ap.parse_args()
    per_font = {ft: [partit(p) for p in sorted(glob.glob(os.path.join(a.dir, ft, "*.npz")))] for ft in FONTS}
    per_font = {k: v for k, v in per_font.items() if v}
    resums = {ft: resum_font(r) for ft, r in per_font.items()}

    det = None
    if a.check:
        iguals, total, dif = 0, 0, []
        for ft, recs in per_font.items():
            for r in recs:
                p2 = os.path.join(a.check, ft, f"{r['partit']}.npz")
                total += 1
                if os.path.exists(p2) and sha256(p2) == r["sha256"]:
                    iguals += 1
                else:
                    dif.append(f"{ft}/{r['partit']}")
        det = {"iguals": iguals, "total": total, "diferents": dif}

    out = ["# Informe de les etiquetes externes", "",
           f"Generat per `bench/report_labels.py` sobre `{a.dir}`. Només agregats de dades externes.",
           "Percentatges de H, A i D sobre frames vàlids. Possessió = tram màxim de H o A dins d'una part; "
           "els frames no vàlids tallen els trams. 'prov.' = mateixa pilota morta, però la possessió crua del "
           "proveïdor sense la regla de duels (IDSSE: BallPossession; SkillCorner: possession.group amb els "
           "nuls omplerts amb l'equip anterior). 'cru' = l'equip provisional que fa servir la regla, sense "
           "aplicar-la (IDSSE: BallPossession; SkillCorner: equip dels player_possession; Metrica: events). "
           "La pilota morta és la mateixa a totes les variants.", ""]
    out += ["## Comprovacions de sincronització", "",
            "- IDSSE: % de `Play` (amb la pilota viva) l'equip dels quals coincideix amb `BallPossession` al "
            "frame sincronitzat. Llindar: 90 %.",
            "- SkillCorner: % de `player_possession` l'equip dels quals coincideix amb el `possession.group` "
            "majoritari (no nul) durant l'event. Queda per sota del 90 %, i per això l'equip provisional surt "
            "dels `player_possession` i no del grup; ara és només informativa.", ""]
    if det:
        out += ["## Determinisme", "",
                f"Dues execucions de `build_labels.py` en directoris diferents: {det['iguals']} de {det['total']} "
                f"fitxers amb el mateix sha256." + (f" Diferents: {', '.join(det['diferents'])}." if det["diferents"] else ""), ""]
    out += ["## Resum per font (mitjanes per partit)", "", "| camp | " + " | ".join(resums) + " |",
            "|---" * (len(resums) + 1) + "|"]
    camps = []
    for s in resums.values():
        camps += [k for k in s if k not in camps]
    for k in camps:
        out.append(f"| {k} | " + " | ".join(f(resums[ft].get(k)) for ft in resums) + " |")
    for ft, recs in per_font.items():
        out += ["", f"## {ft}", ""] + taula_partits(recs)
        extra = [k for k in ("pct_D_pilota_dins", "n_parades_amb_sortida_detectada", "n_represes_sense_obertura",
                             "n_tancades_sense_marca_represa", "n_set_pieces_sense_disparador", "n_accions_controlades")
                 if k in recs[0]]
        if extra:
            out += ["", "| partit | " + " | ".join(extra) + " |", "|---" * (len(extra) + 1) + "|"]
            out += [f"| {r['partit']} | " + " | ".join(f(r.get(k)) for k in extra) + " |" for r in recs]
        if ft == "idsse":
            out += ["", "Desfasament de sincronització per part (ms) i distància mediana Play-pilota (m):", ""]
            for r in recs:
                s = r.get("sincronitzacio") or {}
                out.append(f"- {r['partit']}: " + "; ".join(f"{k} {v['ms']} ms, {v['dist_mediana_m']} m" for k, v in s.items()))
    out += ["", "## sha256", "", "| fitxer | sha256 |", "|---|---|"]
    out += [f"| {ft}/{r['partit']}.npz | `{r['sha256']}` |" for ft, recs in per_font.items() for r in recs]
    with open(a.out, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(json.dumps({"resum": resums, "determinisme": det}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
