"""Inventari de les dades crues externes (Metrica, SkillCorner, IDSSE).

Llegeix /workspace/shared/external_raw/<font>/ i desa bench/inventari_raw.json i
bench/inventari_raw.md. Només dades externes. Cal l'entorn amb kloppy (Metrica Game 3):

    /workspace/shared/venvs/bench/bin/python bench/inventory_raw.py [--raw DIR] [--fonts metrica,skillcorner,idsse]
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import Counter
from xml.etree.ElementTree import iterparse

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))

# events que poden marcar o reiniciar una pilota morta
MORTA = {
    "metrica": re.compile(r"BALL OUT|SET PIECE|FAULT|OFFSIDE|HAND BALL|REFEREE|GOAL|CARD|SUBSTITUTION|INJURY", re.I),
    "skillcorner": re.compile(r"interruption|throw_in|free_kick|corner|goal_kick|kick_off|penalty|out", re.I),
    "idsse": re.compile(r"ThrowIn|FreeKick|CornerKick|GoalKick|KickOff|Penalty|Foul|Offside|Caution|SendingOff|Substitution|FinalWhistle|BallDeflection|OutOf|Referee|Injury|Delete|VAR", re.I),
}


def r2(x):
    return None if x is None else round(float(x), 2)


def pct(n, d):
    return round(100.0 * n / d, 2) if d else None


def trams(v):
    """(valor, llargada) dels trams màxims de valors iguals."""
    v = np.asarray(v)
    if v.size == 0:
        return []
    talls = np.flatnonzero(v[1:] != v[:-1]) + 1
    ini = np.concatenate([[0], talls])
    fi = np.concatenate([talls, [v.size]])
    return [(v[i], int(f - i)) for i, f in zip(ini, fi)]


def durades(ll, fps):
    if not ll:
        return {"n": 0, "mitjana_s": None, "mediana_s": None}
    a = np.asarray(ll, float) / fps
    return {"n": len(a), "mitjana_s": r2(a.mean()), "mediana_s": r2(np.median(a))}


def stats_possessio(seccions, fps):
    """seccions: llista de (equip, viva) per part; equip 0 = cap, 1 = local, 2 = visitant.

    Possessió = tram màxim d'un equip amb la pilota viva (com al concurs)."""
    poss, morta, directes, n_viva, n_tot, morta_amb_equip, n_morta = [], [], 0, 0, 0, 0, 0
    for equip, viva in seccions:
        equip = np.asarray(equip)
        viva = np.asarray(viva, bool)
        n_tot += viva.size
        n_viva += int(viva.sum())
        n_morta += int((~viva).sum())
        morta_amb_equip += int(((~viva) & (equip > 0)).sum())
        estat = np.where(viva, equip, -1)  # -1 = morta
        tr = trams(estat)
        for i, (v, ll) in enumerate(tr):
            if v in (1, 2):
                poss.append(ll)
                if i + 1 < len(tr) and tr[i + 1][0] in (1, 2):
                    directes += 1
            elif v == -1:
                morta.append(ll)
    return {"pct_viva": pct(n_viva, n_tot), "possessions": durades(poss, fps),
            "pilota_morta": durades(morta, fps), "canvis_directes": directes,
            "pct_morta_amb_equip": pct(morta_amb_equip, n_morta),
            "pct_viva_sense_equip": None}


def coords(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
    if xs.size == 0:
        return "desconegut", None, None
    x0, x1 = np.percentile(xs, [1, 99])
    y0, y1 = np.percentile(ys, [1, 99])
    if x1 <= 1.5 and y1 <= 1.5 and x0 >= -0.5:
        s = "normalitzat 0-1"
    elif x0 < -20 and y0 < -10:
        s = "metres centrats"
    elif x0 >= -10 and y0 >= -10:
        s = "metres des d'una cantonada"
    else:
        s = "desconegut"
    return s, [r2(xs.min()), r2(xs.max())], [r2(ys.min()), r2(ys.max())]


def resum_events(tipus, subtipus, font):
    morta = {k: v for k, v in subtipus.items() if MORTA[font].search(k)}
    return {"n_events": sum(tipus.values()), "tipus": dict(sorted(tipus.items())),
            "tipus_subtipus": dict(sorted(subtipus.items())),
            "rellevants_pilota_morta": dict(sorted(morta.items()))}


# ---------------------------------------------------------------- Metrica

def metrica_csv(dirp, nom):
    fitxers = {}
    for costat in ("Home", "Away"):
        fitxers[costat] = os.path.join(dirp, f"{nom}_RawTrackingData_{costat}_Team.csv")
    jug, ball, temps, periode = {}, None, None, None
    for costat, f in fitxers.items():
        with open(f) as fh:
            rd = csv.reader(fh)
            next(rd), next(rd)
            cap = next(rd)
            files = np.array([[float(x) if x not in ("", "NaN") else np.nan for x in r] for r in rd])
        cols = [i for i, c in enumerate(cap) if c.startswith("Player")]
        jug[costat] = np.stack([files[:, [i, i + 1]] for i in cols], axis=1)
        ib = cap.index("Ball")
        if ball is None:
            ball = files[:, [ib, ib + 1]]
            temps, periode = files[:, 2], files[:, 0]
    fps = 1.0 / np.median(np.diff(temps))
    tipus, sub = Counter(), Counter()
    with open(os.path.join(dirp, f"{nom}_RawEventsData.csv")) as fh:
        for r in csv.DictReader(fh):
            tipus[r["Type"]] += 1
            sub[f"{r['Type']} / {r['Subtype'] or '-'}"] += 1
    return jug, ball, fps, periode, tipus, sub, True


def metrica_epts(dirp, nom):
    from kloppy import metrica
    ds = metrica.load_tracking_epts(meta_data=os.path.join(dirp, f"{nom}_metadata.xml"),
                                    raw_data=os.path.join(dirp, f"{nom}_tracking.txt"),
                                    coordinates="metrica")
    df = ds.to_df(engine="pandas")
    jug = {}
    for costat, team in zip(("Home", "Away"), ds.metadata.teams):
        ids = [p.player_id for p in team.players]
        jug[costat] = np.stack([df[[f"{i}_x", f"{i}_y"]].to_numpy(float) for i in ids if f"{i}_x" in df], axis=1)
    ball = df[["ball_x", "ball_y"]].to_numpy(float)
    tipus, sub = Counter(), Counter()
    with open(os.path.join(dirp, f"{nom}_events.json")) as fh:
        for e in json.load(fh)["data"]:
            t = e["type"]["name"]
            st = e.get("subtypes")
            st = [st] if isinstance(st, dict) else (st or [])
            tipus[t] += 1
            sub[f"{t} / {'-'.join(s['name'] for s in st) or '-'}"] += 1
    return jug, ball, float(ds.metadata.frame_rate), df["period_id"].to_numpy(), tipus, sub, True


def inventari_metrica(arrel):
    recs = []
    for dirp in sorted(glob.glob(os.path.join(arrel, "Sample_Game_*"))):
        nom = os.path.basename(dirp)
        if os.path.exists(os.path.join(dirp, f"{nom}_tracking.txt")):
            jug, ball, fps, per, tipus, sub, ids = metrica_epts(dirp, nom)
            fmt = "EPTS (txt + metadata XML) + events JSON"
        else:
            jug, ball, fps, per, tipus, sub, ids = metrica_csv(dirp, nom)
            fmt = "CSV tracking (local/visitant) + CSV events"
        n = len(ball)
        pres = np.isfinite(ball).all(axis=1)
        per_eq = [np.isfinite(jug[c]).all(axis=2).sum(axis=1) for c in ("Home", "Away")]
        tots = (per_eq[0] >= 11) & (per_eq[1] >= 11)
        xs = np.concatenate([ball[:, 0]] + [jug[c][..., 0].ravel() for c in jug])
        ys = np.concatenate([ball[:, 1]] + [jug[c][..., 1].ravel() for c in jug])
        sist, rx, ry = coords(xs, ys)
        recs.append({
            "font": "metrica", "partit": nom, "format": fmt, "fps": r2(fps), "n_frames": n,
            "durada_min": r2(n / fps / 60), "coordenades": sist, "rang_x": rx, "rang_y": ry,
            "alcada_pilota": False, "pct_pilota_present": pct(pres.sum(), n),
            "pct_tots_jugadors": pct(tots.sum(), n), "ids_jugador": ids,
            "ids_jugador_nota": "anonimitzats (PlayerN / P####)",
            "estat_pilota_proveidor": False, "possessio_proveidor": False,
            "events": resum_events(tipus, sub, "metrica"), "possessio": None,
        })
    return recs


# ---------------------------------------------------------------- SkillCorner

def inventari_skillcorner(arrel, ids_dataset):
    recs = []
    for dirp in sorted(glob.glob(os.path.join(arrel, "matches", "*"))):
        mid = os.path.basename(dirp)
        with open(os.path.join(dirp, f"{mid}_match.json")) as fh:
            meta = json.load(fh)
        casa = meta["home_team"]["id"]
        bx, by, bdet, bz, nplay, ndet, grup, per, pid = [], [], [], [], [], [], [], [], []
        xs, ys = [], []
        with open(os.path.join(dirp, f"{mid}_tracking_extrapolated.jsonl")) as fh:
            for i, linia in enumerate(fh):
                d = json.loads(linia)
                b = d.get("ball_data") or {}
                bx.append(np.nan if b.get("x") is None else b["x"])
                by.append(np.nan if b.get("y") is None else b["y"])
                bz.append(b.get("z") is not None)
                bdet.append(bool(b.get("is_detected")))
                pdx = d.get("player_data") or []
                nplay.append(len(pdx))
                ndet.append(sum(1 for p in pdx if p.get("is_detected")))
                if i % 50 == 0:
                    xs += [p["x"] for p in pdx]
                    ys += [p["y"] for p in pdx]
                g = (d.get("possession") or {}).get("group")
                grup.append({"home team": 1, "away team": 2}.get(g, 0))
                pid.append((d.get("possession") or {}).get("player_id") is not None)
                per.append(d.get("period") or 0)
        bx, by, per, grup = np.array(bx), np.array(by), np.array(per), np.array(grup)
        bdet, bz, pid = np.array(bdet), np.array(bz), np.array(pid)
        nplay, ndet = np.array(nplay), np.array(ndet)
        joc = per > 0
        nj = int(joc.sum())
        fps = 10.0
        ts = [p for p in meta.get("match_periods", [])]
        if ts:
            fps = round(ts[0]["duration_frames"] / (ts[0]["duration_minutes"] * 60), 2)
        sist, rx, ry = coords(np.concatenate([bx, xs]), np.concatenate([by, ys]))

        # possessió del proveïdor: grup per frame; sense estat de pilota, tot el període és "viu"
        seccions = [(grup[per == p], np.ones((per == p).sum(), bool)) for p in sorted(set(per[joc]))]
        pos = stats_possessio(seccions, fps)
        pos["pct_viva"] = None
        pos["pct_grup_nul_en_joc"] = pct(((grup == 0) & joc).sum(), nj)
        pos["pct_jugador_en_possessio_en_joc"] = pct((pid & joc).sum(), nj)
        # trams de grup nul (sense reomplir): candidats a pilota morta però no ho són necessàriament
        nuls = [ll for p in sorted(set(per[joc])) for v, ll in trams(grup[per == p]) if v == 0]
        pos["trams_grup_nul"] = durades(nuls, fps)
        pos["nota"] = ("possession.group per frame (home/away/null). No hi ha estat de pilota: "
                       "els trams nuls inclouen pilota morta i joc sense control.")

        tipus, sub, inter = Counter(), Counter(), Counter()
        with open(os.path.join(dirp, f"{mid}_dynamic_events.csv")) as fh:
            for r in csv.DictReader(fh):
                tipus[r["event_type"]] += 1
                sub[f"{r['event_type']} / {r['event_subtype'] or '-'}"] += 1
                for k in ("game_interruption_before", "game_interruption_after", "start_type", "end_type"):
                    if r.get(k):
                        inter[f"{k} = {r[k]}"] += 1
        ev = resum_events(tipus, sub, "skillcorner")
        ev["interrupcions_i_inicis"] = dict(sorted(inter.items()))
        ev["rellevants_pilota_morta"] = dict(sorted(
            {k: v for k, v in inter.items() if k.startswith("game_interruption")}.items()))
        recs.append({
            "font": "skillcorner", "partit": mid, "al_dataset": mid in ids_dataset,
            "competicio": meta["competition_edition"]["name"],
            "format": "JSONL tracking (extrapolat) + match.json + dynamic_events.csv",
            "fps": fps, "n_frames": int(len(per)), "n_frames_en_joc": nj, "durada_min": r2(nj / fps / 60),
            "coordenades": sist, "rang_x": rx, "rang_y": ry, "alcada_pilota": bool(bz[joc].any()),
            "pct_pilota_present": pct((np.isfinite(bx) & joc).sum(), nj),
            "pct_pilota_detectada": pct((bdet & joc).sum(), nj),
            "pct_tots_jugadors": pct(((nplay >= 22) & joc).sum(), nj),
            "jugadors_detectats_mitjans": r2(ndet[joc].mean()),
            "ids_jugador": True, "ids_jugador_nota": "player_id reals (match.json)",
            "estat_pilota_proveidor": False, "possessio_proveidor": True,
            "events": ev, "possessio": pos, "equip_local_id": casa,
        })
    return recs


# ---------------------------------------------------------------- IDSSE (DFL / Sportec)

def idsse_posicions(path):
    ball = {}
    jug = {}
    xs, ys = [], []
    sec = equip = None
    fps = None
    t_prev = None
    ctx = iterparse(path, events=("start", "end"))
    for ev, el in ctx:
        if ev == "start" and el.tag == "FrameSet":
            sec, equip = el.get("GameSection"), el.get("TeamId")
            es_pilota = equip.upper() == "BALL"
            buf = [] if es_pilota else None
            ns = []
        elif ev == "end" and el.tag == "Frame":
            n = int(el.get("N"))
            if es_pilota:
                buf.append((n, float(el.get("X")), float(el.get("Y")), el.get("Z") is not None,
                            int(el.get("BallStatus", -1)), int(el.get("BallPossession", 0) or 0)))
                if fps is None:
                    t = el.get("T")
                    if t_prev is not None and t != t_prev:
                        dt = (np.datetime64(t[:23]) - np.datetime64(t_prev[:23])) / np.timedelta64(1, "ms")
                        fps = round(1000.0 / dt, 2) if dt > 0 else None
                    t_prev = t
            else:
                ns.append(n)
                if n % 250 == 0:
                    xs.append(float(el.get("X")))
                    ys.append(float(el.get("Y")))
            el.clear()
        elif ev == "end" and el.tag == "FrameSet":
            if es_pilota:
                ball[sec] = np.array(buf)
            else:
                jug.setdefault(sec, {}).setdefault(equip, []).append(np.array(ns))
            el.clear()
    return ball, jug, fps, xs, ys


def idsse_events(path):
    tipus, sub = Counter(), Counter()
    for ev, el in iterparse(path, events=("end",)):
        if el.tag != "Event":
            continue
        fills = list(el)
        if fills:
            t = fills[0].tag
            nets = list(fills[0])
            s = "-".join(c.tag for c in nets[:1]) or "-"
            if nets and list(nets[0]):
                s += "/" + list(nets[0])[0].tag
            tipus[t] += 1
            sub[f"{t} / {s}"] += 1
        el.clear()
    return tipus, sub


def idsse_info(path):
    for ev, el in iterparse(path, events=("end",)):
        if el.tag == "General":
            return {k: el.get(k) for k in ("CompetitionName", "Season", "MatchDay", "HomeTeamId", "GuestTeamId")}
    return {}


def inventari_idsse(arrel):
    recs = []
    for pos_f in sorted(glob.glob(os.path.join(arrel, "DFL_04_03_positions_raw_observed_*.xml"))):
        mid = re.search(r"(DFL-MAT-\w+)", pos_f).group(1)
        info = idsse_info(glob.glob(os.path.join(arrel, f"DFL_02_01_matchinformation_*{mid}.xml"))[0])
        ball, jug, fps, xs, ys = idsse_posicions(pos_f)
        fps = fps or 25.0
        seccions, n, pres, tots = [], 0, 0, 0
        bxs, bys, zz = [], [], False
        valors_estat, valors_poss = Counter(), Counter()
        for sec in sorted(ball):
            b = ball[sec]
            nmin, nmax = int(b[:, 0].min()), int(b[:, 0].max())
            span = nmax - nmin + 1
            n += span
            pres += len(np.unique(b[:, 0]))
            bxs.append(b[:, 1])
            bys.append(b[:, 2])
            zz = zz or bool(b[:, 3].any())
            valors_estat.update(b[:, 4].astype(int).tolist())
            valors_poss.update(b[:, 5].astype(int).tolist())
            comptes = []
            for eq, llistes in sorted(jug.get(sec, {}).items()):
                if eq.lower() == "referee":
                    continue
                c = np.zeros(span, int)
                for ns in llistes:
                    ns = ns[(ns >= nmin) & (ns <= nmax)] - nmin
                    c += np.bincount(ns, minlength=span)
                comptes.append(c)
            if len(comptes) >= 2:
                tots += int(((comptes[0] >= 11) & (comptes[1] >= 11)).sum())
            ordre = np.argsort(b[:, 0], kind="stable")
            seccions.append((b[ordre, 5].astype(int), b[ordre, 4] == 1))
        sist, rx, ry = coords(np.concatenate(bxs + [np.array(xs)]), np.concatenate(bys + [np.array(ys)]))
        pos = stats_possessio(seccions, fps)
        viva_sense = sum(int(((e == 0) & v).sum()) for e, v in seccions)
        pos["pct_viva_sense_equip"] = pct(viva_sense, sum(int(v.sum()) for _, v in seccions))
        pos["valors_BallStatus"] = {str(k): v for k, v in sorted(valors_estat.items())}
        pos["valors_BallPossession"] = {str(k): v for k, v in sorted(valors_poss.items())}
        pos["nota"] = "BallStatus (0 morta, 1 viva) i BallPossession (1 local, 2 visitant) per frame al FrameSet BALL."
        ev_f = glob.glob(os.path.join(arrel, f"DFL_03_02_events_raw_*{mid}.xml"))
        tipus, sub = idsse_events(ev_f[0]) if ev_f else (Counter(), Counter())
        recs.append({
            "font": "idsse", "partit": mid, "competicio": f"{info.get('CompetitionName')} {info.get('Season')}",
            "format": "XML DFL (positions_raw_observed, events_raw, matchinformation)",
            "fps": fps, "n_frames": n, "durada_min": r2(n / fps / 60), "coordenades": sist,
            "rang_x": rx, "rang_y": ry, "alcada_pilota": zz,
            "pct_pilota_present": pct(pres, n), "pct_tots_jugadors": pct(tots, n),
            "ids_jugador": True, "ids_jugador_nota": "PersonId DFL-OBJ-*, TeamId DFL-CLU-*",
            "estat_pilota_proveidor": bool(set(valors_estat) & {0, 1}),
            "possessio_proveidor": bool(set(valors_poss) & {1, 2}),
            "events": resum_events(tipus, sub, "idsse"), "possessio": pos,
            "nota_events": "events amb EventTime de rellotge, no amb número de frame",
        })
    return recs


# ---------------------------------------------------------------- sortida

def resum(recs):
    def rang(f):
        v = [x for x in (f(r) for r in recs) if x is not None]
        return f"{min(v):g}-{max(v):g} (mitj. {np.mean(v):.4g})" if v else "-"
    s = {
        "partits": len(recs), "fps": sorted({r["fps"] for r in recs}),
        "coordenades": sorted({r["coordenades"] for r in recs}),
        "alcada_pilota": any(r["alcada_pilota"] for r in recs),
        "durada_min": rang(lambda r: r["durada_min"]),
        "pct_pilota_present": rang(lambda r: r["pct_pilota_present"]),
        "pct_tots_jugadors": rang(lambda r: r["pct_tots_jugadors"]),
        "ids_jugador": all(r["ids_jugador"] for r in recs),
        "estat_pilota_proveidor": any(r["estat_pilota_proveidor"] for r in recs),
        "possessio_proveidor": any(r["possessio_proveidor"] for r in recs),
        "events": all(r["events"]["n_events"] > 0 for r in recs),
    }
    if recs and recs[0].get("pct_pilota_detectada") is not None:
        s["pct_pilota_detectada"] = rang(lambda r: r.get("pct_pilota_detectada"))
    if any(r["possessio"] for r in recs):
        s["pct_viva"] = rang(lambda r: r["possessio"]["pct_viva"])
        s["possessions"] = rang(lambda r: r["possessio"]["possessions"]["n"])
        s["possessio_mitjana_s"] = rang(lambda r: r["possessio"]["possessions"]["mitjana_s"])
        s["possessio_mediana_s"] = rang(lambda r: r["possessio"]["possessions"]["mediana_s"])
        s["morta_mitjana_s"] = rang(lambda r: r["possessio"]["pilota_morta"]["mitjana_s"])
        s["morta_mediana_s"] = rang(lambda r: r["possessio"]["pilota_morta"]["mediana_s"])
        s["canvis_directes"] = rang(lambda r: r["possessio"]["canvis_directes"])
        s["pct_morta_amb_equip"] = rang(lambda r: r["possessio"]["pct_morta_amb_equip"])
        s["pct_viva_sense_equip"] = rang(lambda r: r["possessio"].get("pct_viva_sense_equip"))
        s["pct_grup_nul_en_joc"] = rang(lambda r: r["possessio"].get("pct_grup_nul_en_joc"))
    return s


def markdown(per_font, resums):
    f = lambda x: "-" if x is None else (f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))
    out = ["# Inventari de les dades crues externes", "",
           "Generat per `bench/inventory_raw.py` sobre `/workspace/shared/external_raw/`. Només dades externes.",
           "Possessió del proveïdor: tram màxim d'un equip amb la pilota viva, separat per parts.",
           "'% 22 jug.' < 100 a IDSSE vol dir expulsions (un equip amb 10). A SkillCorner, el tracking "
           "extrapolat porta sempre els 22 o cap, i per això coincideix amb '% pilota'. A SkillCorner "
           "no hi ha estat de pilota: la possessió són trams de `possession.group` no nul.", ""]
    out += ["## Resum per font", "", "| camp | " + " | ".join(resums) + " |", "|---" * (len(resums) + 1) + "|"]
    camps = sorted({k for s in resums.values() for k in s}, key=lambda k: list(next(iter(resums.values()))).index(k)
                   if k in next(iter(resums.values())) else 99)
    for k in camps:
        out.append(f"| {k} | " + " | ".join(f(resums[ft].get(k)) for ft in resums) + " |")
    for font, recs in per_font.items():
        out += ["", f"## {font}", "",
                "| partit | fps | frames | min | coord. | % pilota | % 22 jug. | IDs | estat pilota | possessió | events | % viva | poss. | poss. mitj./med. s | morta mitj./med. s |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in recs:
            p = r["possessio"] or {}
            ps, mo = p.get("possessions") or {}, p.get("pilota_morta") or {}
            out.append("| " + " | ".join([
                r["partit"], f(r["fps"]), f(r["n_frames"]), f(r["durada_min"]), r["coordenades"],
                f(r["pct_pilota_present"]), f(r["pct_tots_jugadors"]), "sí" if r["ids_jugador"] else "no",
                "sí" if r["estat_pilota_proveidor"] else "no", "sí" if r["possessio_proveidor"] else "no",
                f(r["events"]["n_events"]), f(p.get("pct_viva")), f(ps.get("n")),
                f"{f(ps.get('mitjana_s'))} / {f(ps.get('mediana_s'))}",
                f"{f(mo.get('mitjana_s'))} / {f(mo.get('mediana_s'))}"]) + " |")
        tot, rel = Counter(), Counter()
        for r in recs:
            tot.update(r["events"]["tipus"])
            rel.update(r["events"]["rellevants_pilota_morta"])
        out += ["", f"### Tipus d'event ({font}, suma de partits)", "", "| tipus | n |", "|---|---|"]
        out += [f"| `{k}` | {v} |" for k, v in sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))]
        out += ["", f"### Events rellevants per a pilota morta ({font})", "", "| tipus / subtipus | n |", "|---|---|"]
        out += [f"| `{k}` | {v} |" for k, v in sorted(rel.items(), key=lambda kv: (-kv[1], kv[0]))]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/workspace/shared/external_raw")
    ap.add_argument("--dataset", default="/workspace/shared/dataset")
    ap.add_argument("--fonts", default="metrica,skillcorner,idsse")
    ap.add_argument("--out", default=AQUI)
    a = ap.parse_args()
    ids = {m.group(1) for p in glob.glob(os.path.join(a.dataset, "skillcorner_*.npz"))
           if (m := re.search(r"skillcorner_(\d+)", p))}
    per_font = {}
    for font in a.fonts.split(","):
        arrel = os.path.join(a.raw, font)
        if font == "metrica":
            per_font[font] = inventari_metrica(arrel)
        elif font == "skillcorner":
            per_font[font] = inventari_skillcorner(arrel, ids)
        elif font == "idsse":
            per_font[font] = inventari_idsse(arrel)
        print(font, len(per_font[font]), "partits", flush=True)
    resums = {ft: resum(r) for ft, r in per_font.items()}
    with open(os.path.join(a.out, "inventari_raw.json"), "w") as fh:
        json.dump({"partits": per_font, "resum_per_font": resums}, fh, indent=1, ensure_ascii=False, sort_keys=True)
    with open(os.path.join(a.out, "inventari_raw.md"), "w") as fh:
        fh.write(markdown(per_font, resums))
    print(json.dumps(resums, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
