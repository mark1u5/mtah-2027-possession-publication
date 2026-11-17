"""Paràmetres de la degradació v3 (C2c): forats de pilota d'IDSSE segons si la pilota és viva o morta.

    P=/workspace/shared/venvs/bench/bin/python
    $P bench/forats_v3.py mesura     # SkillCorner sense degradar: r = taxa(D) / taxa(viva), durades per estat
                                     # i % de frames de forat en D (net i amb l'ompliment de la v2)
    $P bench/forats_v3.py calibra    # k_viva i k_D a IDSSE (llavor 0): % absent de l'objectiu i % en D de SkillCorner

Les taxes base per cel·la i les durades són les de la v2 (target_stats.json). De SkillCorner, que és
només d'entrenament, en surt la referència del % en D; r i les durades per estat s'informen però no
entren a la degradació. Metrica no s'hi fa servir. Desa bench/forats_v3.json, només amb agregats.
"""
import argparse
import glob
import json
import os
import sys
import zlib

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import tracking_stats as ts  # noqa: E402

ORIGINALS = "/workspace/shared/bench_v7"
SORTIDA = os.path.join(AQUI, "forats_v3.json")
TARGET = os.path.join(AQUI, "target_stats.json")


def carrega_net(path):
    """Tracking sense degradar, només amb les deteccions reals."""
    z = np.load(path, allow_pickle=False)
    b = z["ball_xyz"][:, :2].copy()
    b[~z["ball_detected"]] = np.nan
    xy = z["players_xy"].copy()
    xy[~z["players_detected"]] = np.nan
    lab = z["label_acc"].astype("<U1")
    meta = json.loads(str(z["meta"]))
    return z, b, xy, lab, tuple(meta["camp_m"]), meta


def pct_forat_D(b, lab, z):
    ok = (z["period"] > 0) & (z["valid"] == 1) & np.isin(lab, ("H", "A", "D"))
    f = ok & ~np.isfinite(b[:, 0])
    return int(f.sum()), int((f & (lab == "D")).sum())


def marginals(cel):
    out = {}
    for pos, franges in ((0, ts.VEL_CLAUS[:3]), (1, ts.LINIA_FRANGES), (2, ts.JUG_FRANGES)):
        for fr in franges:
            s = {e: [0, 0] for e in ("viva", "D")}
            for k, c in cel.items():
                if k.split("|")[pos] == fr:
                    for e in s:
                        s[e][0] += c[e]["inicis"]
                        s[e][1] += c[e]["oportunitats"]
            tv = s["viva"][0] / s["viva"][1] if s["viva"][1] else None
            td = s["D"][0] / s["D"][1] if s["D"][1] else None
            out[fr] = {"taxa_viva": tv, "taxa_D": td, "r": td / tv if tv and td is not None else None,
                       "inicis_viva": s["viva"][0], "inicis_D": s["D"][0]}
    return out


def cmd_mesura(a):
    import degrade as dg
    with open(TARGET) as fh:
        target = json.load(fh)
    franja = dg.franja_buida(target)
    cel, durs = None, {"viva": [], "D": []}
    forat = {"net": [0, 0], "omplert": [0, 0]}
    for path in sorted(glob.glob(os.path.join(ORIGINALS, "skillcorner", "*.npz"))):
        z, b, xy, lab, camp, _ = carrega_net(path)
        mask = (z["period"] > 0) & (z["valid"] == 1) & np.isin(lab, ("H", "A", "D"))
        c, d = ts.stats_estat(b, xy, camp, mask, lab == "D")
        if cel is None:
            cel = c
        else:
            for k in cel:
                for e in ("viva", "D"):
                    for q in ("inicis", "oportunitats"):
                        cel[k][e][q] += c[k][e][q]
        for e in durs:
            durs[e].append(d[e])
        for nom, bb in (("net", b), ("omplert", dg.omple_franja(b, z["period"], franja)[0])):
            n, nd = pct_forat_D(bb, lab, z)
            forat[nom][0] += n
            forat[nom][1] += nd
    agr = ts.agrupa_estat(cel)
    durs = {e: ts.distribucio(np.concatenate(v)) for e, v in durs.items()}
    tot = {e: sum(c[e]["inicis"] for c in cel.values()) / max(1, sum(c[e]["oportunitats"] for c in cel.values()))
           for e in ("viva", "D")}
    rs = [v["r"] for v in agr.values() if v["r"] is not None]
    out = {"font": "skillcorner (sense degradar, frames vàlids, label_acc)", "cel": cel, "r_agrupat": agr,
           "durades": durs, "franja_omplerta_frames": list(franja),
           "resum": {"taxa_viva": tot["viva"], "taxa_D": tot["D"], "r_global": tot["D"] / tot["viva"],
                     "r_cel·les": {"min": min(rs), "mediana": float(np.median(rs)), "max": max(rs)},
                     "grups": {g: sum(1 for v in agr.values() if v["grup"] == g) for g in sorted({v["grup"] for v in agr.values()})},
                     "marginals": marginals(cel),
                     "pct_forat_en_D_net": round(100.0 * forat["net"][1] / forat["net"][0], 2),
                     "pct_forat_en_D_omplert": round(100.0 * forat["omplert"][1] / forat["omplert"][0], 2),
                     "durada_mediana_s": {e: durs[e]["quantils_s"]["p50"] for e in durs}}}
    with open(SORTIDA, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, sort_keys=True)
    r = out["resum"]
    print(json.dumps({k: r[k] for k in ("taxa_viva", "taxa_D", "r_global", "r_cel·les", "grups",
                                        "pct_forat_en_D_net", "pct_forat_en_D_omplert", "durada_mediana_s")},
                     ensure_ascii=False, indent=1))
    for fr, m in r["marginals"].items():
        print(f"  {fr:8s} r={m['r'] if m['r'] is None else round(m['r'], 2)}  inicis viva/D {m['inicis_viva']}/{m['inicis_D']}")


def cmd_calibra(a):
    """k_viva i k_D tals que IDSSE degradat (llavor de calibratge, després de l'ompliment) fa el %
    absent de l'objectiu i el % de frames de forat en D de SkillCorner amb el mateix ompliment.
    Bisecció niada: per fora, el quocient k_D / k_viva (mou el % en D); per dins, l'escala (mou el %
    absent)."""
    import degrade as dg
    with open(TARGET) as fh:
        target = json.load(fh)
    with open(SORTIDA) as fh:
        p3 = json.load(fh)
    franja = dg.franja_buida(target)
    partits = []
    for path in sorted(glob.glob(os.path.join(ORIGINALS, "idsse", "*.npz"))):
        z, b, xy, lab, camp, meta = carrega_net(path)
        partits.append((z, b, dg.senyals_v3(b, xy, camp), lab, meta))

    def mesura(kv, kd):
        n = na = nf = nfd = 0
        forats = []
        for z, b, sen, lab, meta in partits:
            # el mateix generador que degrade.py amb aquesta llavor
            rng = np.random.default_rng([a.llavor, zlib.crc32(f"{meta['font']}/{meta['partit']}".encode())])
            b2, _ = dg.forats_pilota_v3(rng, b, z["period"], lab == "D", target, kv, kd, sen)
            b2, _ = dg.omple_franja(b2, z["period"], franja)
            m = z["period"] > 0
            ab = m & ~np.isfinite(b2[:, 0])
            n += int(m.sum())
            na += int(ab.sum())
            f, fd = pct_forat_D(b2, lab, z)
            nf += f
            nfd += fd
            forats += [e - s for s, e in ts.trams(ab)]
        return 100.0 * na / n, 100.0 * nfd / max(nf, 1), float(np.median(forats)) / ts.FPS

    obj_abs = target["pilota"]["pct_absent"]
    obj_D = p3["resum"]["pct_forat_en_D_omplert"]

    def escala(rho):
        lo, hi = np.log(1e-6), np.log(1e3)
        for _ in range(a.iteracions):
            mig = 0.5 * (lo + hi)
            if mesura(float(np.exp(mig)), float(np.exp(mig)) * rho)[0] < obj_abs:
                lo = mig
            else:
                hi = mig
        kv = float(np.exp(0.5 * (lo + hi)))
        return kv, mesura(kv, kv * rho)

    passos = []
    lo, hi = np.log(1e-2), np.log(1e6)
    for _ in range(a.iteracions):
        mig = 0.5 * (lo + hi)
        kv, (ab, pd, med) = escala(float(np.exp(mig)))
        passos.append({"rho": round(float(np.exp(mig)), 4), "k_viva": round(kv, 6), "absent": round(ab, 3),
                       "forat_D": round(pd, 3), "mediana_s": round(med, 3)})
        print(json.dumps(passos[-1]), flush=True)
        if pd < obj_D:
            lo = mig
        else:
            hi = mig
    rho = float(np.exp(0.5 * (lo + hi)))
    kv, (ab, pd, med) = escala(rho)
    p3.update({"k_viva": kv, "k_D": kv * rho})
    p3.pop("alfa", None)
    p3["calibratge"] = {"objectiu_pct_absent": obj_abs, "objectiu_pct_forat_D": obj_D, "llavor": a.llavor,
                        "iteracions": a.iteracions, "pct_absent": round(ab, 3), "pct_forat_D": round(pd, 3),
                        "mediana_s": round(med, 3), "passos": passos}
    with open(SORTIDA, "w") as fh:
        json.dump(p3, fh, indent=1, ensure_ascii=False, sort_keys=True)
    print(json.dumps({"k_viva": kv, "k_D": kv * rho, **{k: v for k, v in p3["calibratge"].items() if k != "passos"}}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ordre", choices=["mesura", "calibra"])
    ap.add_argument("--llavor", type=int, default=0)
    ap.add_argument("--iteracions", type=int, default=14)
    a = ap.parse_args()
    {"mesura": cmd_mesura, "calibra": cmd_calibra}[a.ordre](a)


if __name__ == "__main__":
    main()
