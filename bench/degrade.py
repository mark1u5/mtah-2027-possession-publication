"""Degrada els partits externs perquè s'assemblin al tracking del partit proporcionat.

Paràmetres: bench/target_stats.json (agregats del tracking del partit, sense GT). Cap cerca ni
optimització. Les etiquetes i valid no es toquen.

    # v2 (forats d'IDSSE condicionats a la línia i als jugadors); SkillCorner i Metrica no canvien
    /workspace/shared/venvs/bench/bin/python bench/degrade.py --in /workspace/shared/bench_v7 \
        --out /workspace/shared/bench_v7b_deg --seeds 0 1 2 --fonts idsse
    # v1, la de C1 a C3
    ... --out /workspace/shared/bench_v7_deg --versio v1
    # v3 (C2c): la v2 amb un multiplicador per a la pilota viva i un per a la morta (bench/forats_v3.py)
    ... --out /workspace/shared/bench_v7d_deg --fonts idsse --versio v3
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
from build_labels import desa_npz  # noqa: E402

PHI = 0.9            # AR(1) del soroll: temps de correlació -1/ln(0,9) ≈ 9,5 frames (0,38 s)
DESPL_REPRESA = (-0.5, 1.0)  # s, per definició: la pilota reapareix poc abans o poc després del primer toc
SOROLL_CLAR = 0.8    # SkillCorner/Metrica: només s'afegeix soroll si el propi < 0,8 × objectiu
CLAUS_ETIQUETA = ["label_acc", "label_dur", "label_cru", "label_prov", "valid", "period", "frame", "frame_font"]


def mostra_quantils(rng, q101, n=None):
    """Mostra de la distribució donada per la funció de quantils 0..100 (interpolació lineal)."""
    u = rng.uniform(0, 100, size=n)
    return np.interp(u, np.arange(101), np.asarray(q101, float))


def forats_pilota(rng, bola, periode, target):
    """Forats a la pilota amb la taxa d'inici i les durades condicionades a la velocitat."""
    cond = target["pilota"]["inici_condicionat"]
    n = len(bola)
    absent = ~np.isfinite(bola[:, 0])
    vel = np.full(n, np.nan)
    vel[1:] = np.hypot(np.diff(bola[:, 0]), np.diff(bola[:, 1])) * ts.FPS
    u = rng.uniform(size=n)
    out = absent.copy()
    t = 1
    n_forats = 0
    while t < n - 1:
        if periode[t] <= 0 or out[t] or out[t - 1] or periode[t + 1] != periode[t]:
            t += 1
            continue
        k = ts.clau_condicio(vel[t], None)
        c = cond.get(k)
        if c and u[t] < c["taxa_inici"] and c["durades"].get("n", 0) > 0:
            d = max(1, int(round(float(mostra_quantils(rng, c["durades"]["quantils101_frames"])))))
            fi = t + 1 + d
            # el forat no travessa el canvi de part
            fi = min(fi, int(np.flatnonzero(periode == periode[t])[-1]) + 1)
            out[t + 1:fi] = True
            n_forats += 1
            t = fi
        else:
            t += 1
    b = bola.copy()
    b[out] = np.nan
    return b, n_forats


def forats_pilota_v2(rng, bola, xy, periode, camp, target):
    """Com forats_pilota, però la taxa d'inici depèn de la velocitat de la pilota, de la distància a
    la línia més propera i de la velocitat mitjana dels jugadors (grups de tracking_stats.agrupa_v2),
    i la durada, de la velocitat mitjana dels jugadors."""
    v2 = target["pilota"]["inici_v2"]
    taxes, durs = v2["taxes_agrupades"], v2["durades_agrupades"]
    n = len(bola)
    absent = ~np.isfinite(bola[:, 0])
    vel = np.full(n, np.nan)
    vel[1:] = np.hypot(np.diff(bola[:, 0]), np.diff(bola[:, 1])) * ts.FPS
    with np.errstate(invalid="ignore"):
        li = ts.franja_linia(bola[:, 0], bola[:, 1], camp)
    ju = ts.franja_jug(ts.vel_jugadors(xy))
    u = rng.uniform(size=n)
    out = absent.copy()
    t = 1
    n_forats = 0
    while t < n - 1:
        if periode[t] <= 0 or out[t] or out[t - 1] or periode[t + 1] != periode[t]:
            t += 1
            continue
        j = ts.JUG_FRANGES[ju[t]]
        k = f"{ts.clau_condicio(vel[t], None)}|{ts.LINIA_FRANGES[li[t]]}|{j}"
        if u[t] < taxes[k]["taxa"] and durs[j].get("n", 0) > 0:
            d = max(1, int(round(float(mostra_quantils(rng, durs[j]["quantils101_frames"])))))
            fi = min(t + 1 + d, int(np.flatnonzero(periode == periode[t])[-1]) + 1)
            out[t + 1:fi] = True
            n_forats += 1
            t = fi
        else:
            t += 1
    b = bola.copy()
    b[out] = np.nan
    return b, n_forats


def senyals_v3(bola, xy, camp):
    """Senyals de la v2 per frame: velocitat de la pilota, franja de línia i franja de jugadors."""
    n = len(bola)
    vel = np.full(n, np.nan)
    vel[1:] = np.hypot(np.diff(bola[:, 0]), np.diff(bola[:, 1])) * ts.FPS
    with np.errstate(invalid="ignore"):
        li = ts.franja_linia(bola[:, 0], bola[:, 1], camp)
    ju = ts.franja_jug(ts.vel_jugadors(xy))
    claus = [f"{ts.clau_condicio(vel[t], None)}|{ts.LINIA_FRANGES[li[t]]}|{ts.JUG_FRANGES[ju[t]]}" for t in range(n)]
    return claus


def forats_pilota_v3(rng, bola, periode, morta, target, k_viva, k_D, claus):
    """Com la v2 (mateixa taxa base per cel·la i mateixes durades per franja de jugadors), però la
    taxa es multiplica per k_viva o per k_D segons si la pilota és viva o morta (label_acc). Un
    forat que comença amb la pilota morta acaba, com a molt, a la represa més un desplaçament
    uniforme de DESPL_REPRESA segons."""
    v2 = target["pilota"]["inici_v2"]
    taxes, durs = v2["taxes_agrupades"], v2["durades_agrupades"]
    n = len(bola)
    out = ~np.isfinite(bola[:, 0])
    u = rng.uniform(size=n)
    fi_part = {p: int(np.flatnonzero(periode == p)[-1]) + 1 for p in np.unique(periode) if p > 0}
    # primer frame de joc viu a partir de cada frame (la represa del tram de D)
    represa = np.full(n, n)
    viu = np.flatnonzero(~np.asarray(morta, bool))
    pos = np.searchsorted(viu, np.arange(n))
    ok = pos < viu.size
    represa[ok] = viu[pos[ok]]
    t = 1
    n_forats = 0
    while t < n - 1:
        if periode[t] <= 0 or out[t] or out[t - 1] or periode[t + 1] != periode[t]:
            t += 1
            continue
        k = claus[t]
        j = k.rsplit("|", 1)[1]
        if u[t] < taxes[k]["taxa"] * (k_D if morta[t] else k_viva) and durs[j].get("n", 0) > 0:
            d = max(1, int(round(float(mostra_quantils(rng, durs[j]["quantils101_frames"])))))
            fi = min(t + 1 + d, fi_part[periode[t]])
            if morta[t]:
                # la pilota torna a la represa, una mica abans o després del primer toc
                desp = int(round(float(rng.uniform(*DESPL_REPRESA)) * ts.FPS))
                fi = max(t + 2, min(fi, int(represa[t]) + desp))
            out[t + 1:fi] = True
            n_forats += 1
            t = fi
        else:
            t += 1
    b = bola.copy()
    b[out] = np.nan
    return b, n_forats


def franja_buida(target, pct_max=1.0):
    """Franja de durades de forat que l'objectiu gairebé no té: la tirada més llarga de franges
    logarítmiques amb <= pct_max % dels forats, per sota del p25. Torna (min, max) en frames."""
    h = target["pilota"]["forats"]["hist_log_frames"]
    vores, rec = h["vores"], h["recomptes"]
    total = sum(rec)
    p25 = target["pilota"]["forats"]["quantils_s"]["p25"] * ts.FPS
    millor, actual = None, None
    for i, (v, c) in enumerate(zip(vores, rec)):
        dins = 2 * v <= p25 and 100.0 * c / total <= pct_max
        if dins:
            actual = (actual[0], i) if actual else (i, i)
            if millor is None or actual[1] - actual[0] > millor[1] - millor[0]:
                millor = actual
        else:
            actual = None
    if millor is None:
        return None
    return vores[millor[0]], 2 * vores[millor[1]]


def omple_franja(bola, periode, franja):
    """Interpola linealment els forats de pilota amb durada dins de la franja."""
    if franja is None:
        return bola, 0
    lo, hi = franja
    b = bola.copy()
    n_omplerts = 0
    absent = ~np.isfinite(b[:, 0])
    for a, e in ts.trams(absent):
        d = e - a
        if not lo <= d < hi or a == 0 or e >= len(b) or periode[a - 1] != periode[e] or periode[e] <= 0:
            continue
        w = (np.arange(1, d + 1) / (d + 1))[:, None]
        for c in range(b.shape[1]):
            if np.isfinite(b[a - 1, c]) and np.isfinite(b[e, c]):
                b[a:e, c] = b[a - 1, c] + (b[e, c] - b[a - 1, c]) * w[:, 0]
        n_omplerts += 1
    return b, n_omplerts


def fragmenta(rng, xy, equip, periode, target, p_manté):
    """Talla cada trajectòria en fragments i els reparteix en slots anònims per equip."""
    qf = target["jugadors"]["fragments"]["quantils101_frames"]
    mitj_frag = target["jugadors"]["fragments"]["mitjana_s"] * ts.FPS
    mitj_buit = mitj_frag * (1 - p_manté) / p_manté if p_manté < 1 else 0.0
    frags = {0: [], 1: []}
    for s in range(xy.shape[1]):
        pres = np.isfinite(xy[:, s, 0]) & (periode > 0)
        for a, b in ts.trams(pres):
            pos = a
            while pos < b:
                ll = max(1, int(round(float(mostra_quantils(rng, qf)))))
                fi = min(pos + ll, b)
                frags[int(equip[s])].append((pos, fi, s))
                buit = int(rng.geometric(1.0 / (1.0 + mitj_buit))) - 1 if mitj_buit > 0 else 0
                pos = fi + buit
    n = xy.shape[0]
    columnes, equips = [], []
    for e in (0, 1):
        ultim = []  # últim frame ocupat de cada slot d'aquest equip
        slot_de = []
        for pos, fi, s in sorted(frags[e]):
            lliures = [i for i, u in enumerate(ultim) if u < pos - 1]  # com a mínim un frame buit
            if lliures:
                i = lliures[int(rng.integers(len(lliures)))]
            else:
                ultim.append(-10)
                i = len(ultim) - 1
            ultim[i] = fi - 1
            slot_de.append((i, pos, fi, s))
        base = len(columnes)
        columnes += [None] * len(ultim)
        equips += [e] * len(ultim)
        for i, pos, fi, s in slot_de:
            if columnes[base + i] is None:
                columnes[base + i] = []
            columnes[base + i].append((pos, fi, s))
    out = np.full((n, len(columnes), 2), np.nan, np.float32)
    for c, llista in enumerate(columnes):
        for pos, fi, s in llista or []:
            out[pos:fi, c] = xy[pos:fi, s]
    return out, np.array(equips, np.int8), sum(len(v) for v in frags.values())


def soroll_ar1(rng, pres, sigma):
    """Soroll AR(1) amb desviació marginal sigma, reiniciat a l'inici de cada tram present."""
    shape = pres.shape + (2,)
    w = rng.normal(size=shape) * sigma
    e = np.zeros(shape)
    k = np.sqrt(1 - PHI ** 2)
    prev_pres = np.zeros(pres.shape[1:], bool)
    for t in range(pres.shape[0]):
        nou = pres[t] & ~prev_pres
        e[t] = np.where(nou[..., None], w[t], PHI * e[t - 1] + k * w[t]) if t else w[t]
        prev_pres = pres[t]
    e[~pres] = 0
    return e


def degrada(path, seed, target, out_dir, versio="v2", p3=None):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    font, partit = meta["font"], meta["partit"]
    rng = np.random.default_rng([seed, zlib.crc32(f"{font}/{partit}".encode())])
    periode = z["period"]
    bola = z["ball_xyz"].copy()
    xy = z["players_xy"].copy()
    equip = z["players_team"]
    # només deteccions reals: les posicions extrapolades del proveïdor no compten
    bola[~z["ball_detected"]] = np.nan
    xy[~z["players_detected"]] = np.nan
    mask = periode > 0
    abans = ts.estadistiques(bola[:, :2], xy, equip, mask)
    info = {"llavor": seed, "font": font, "partit": partit, "versio_forats": versio}

    n_forats = 0
    if font == "idsse" and versio == "v3":
        claus = senyals_v3(bola[:, :2], xy, meta["camp_m"])
        morta = z["label_acc"].astype("<U1") == "D"
        bola2, n_forats = forats_pilota_v3(rng, bola[:, :2], periode, morta, target, p3["k_viva"], p3["k_D"], claus)
        bola[~np.isfinite(bola2[:, 0])] = np.nan
        info.update({"k_viva": p3["k_viva"], "k_D": p3["k_D"]})
    elif font == "idsse" and versio == "v2":
        bola2, n_forats = forats_pilota_v2(rng, bola[:, :2], xy, periode, meta["camp_m"], target)
        bola[~np.isfinite(bola2[:, 0])] = np.nan
    elif font == "idsse":
        bola2, n_forats = forats_pilota(rng, bola[:, :2], periode, target)
        bola[~np.isfinite(bola2[:, 0])] = np.nan
    info["forats_pilota_afegits"] = n_forats
    # el tracker del partit gairebé no deixa forats en aquesta franja: els omplim (canvi de model, v2)
    franja = franja_buida(target)
    bola, n_omplerts = omple_franja(bola, periode, franja)
    info.update({"franja_omplerta_frames": list(franja) if franja else None, "forats_omplerts": n_omplerts})

    vis_obj = target["jugadors"]["visibles_mitjana"]
    vis_propi = abans["jugadors"]["visibles_mitjana"]
    p = min(1.0, vis_obj / vis_propi) if vis_propi > 0 else 1.0
    xy2, equip2, n_frag = fragmenta(rng, xy, equip, periode, target, p)
    info.update({"p_mante": round(p, 4), "fragments": n_frag, "slots": int(xy2.shape[1])})

    obj_j, obj_b = target["soroll"]["jugadors_std_m"], target["soroll"]["pilota_std_m"]
    prop_j, prop_b = abans["soroll"]["jugadors_std_m"] or 0.0, abans["soroll"]["pilota_std_m"] or 0.0
    afegeix = font == "idsse" or prop_j < SOROLL_CLAR * obj_j
    sj = float(np.sqrt(max(0.0, obj_j ** 2 - prop_j ** 2))) if afegeix else 0.0
    sb = float(np.sqrt(max(0.0, obj_b ** 2 - prop_b ** 2))) if afegeix else 0.0
    if sj > 0:
        xy2 = (xy2 + soroll_ar1(rng, np.isfinite(xy2[..., 0]), sj)).astype(np.float32)
    if sb > 0:
        pb = np.isfinite(bola[:, 0])
        bola[:, :2] = (bola[:, :2] + soroll_ar1(rng, pb[:, None], sb)[:, 0]).astype(np.float32)
    info.update({"soroll_jugadors_sigma_m": round(sj, 6), "soroll_pilota_sigma_m": round(sb, 6),
                 "soroll_propi_jugadors_m": prop_j, "soroll_propi_pilota_m": prop_b})

    d = {k: z[k] for k in CLAUS_ETIQUETA}
    d.update({"ball_xyz": bola.astype(np.float32), "players_xy": xy2, "players_team": equip2,
              "meta": z["meta"], "deg_meta": np.array(json.dumps(info, sort_keys=True))})
    desti = os.path.join(out_dir, font, f"{partit}_s{seed}.npz")
    desa_npz(desti, d)
    return desti, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/workspace/shared/bench_v7")
    ap.add_argument("--out", default="/workspace/shared/bench_v7_deg")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--target", default=os.path.join(AQUI, "target_stats.json"))
    ap.add_argument("--versio", default="v2", choices=["v1", "v2", "v3"], help="model dels forats de pilota d'IDSSE")
    ap.add_argument("--params-v3", default=os.path.join(AQUI, "forats_v3.json"), help="sortida de forats_v3.py")
    ap.add_argument("--fonts", nargs="+", default=["idsse", "skillcorner", "metrica"])
    a = ap.parse_args()
    with open(a.target) as fh:
        target = json.load(fh)
    p3 = None
    if a.versio == "v3":
        with open(a.params_v3) as fh:
            p3 = json.load(fh)
    for path in sorted(glob.glob(os.path.join(a.inp, "*", "*.npz"))):
        if os.path.basename(os.path.dirname(path)) not in a.fonts:
            continue
        for seed in a.seeds:
            desti, info = degrada(path, seed, target, a.out, a.versio, p3)
            print(os.path.relpath(desti, a.out), f"forats={info['forats_pilota_afegits']} "
                  f"p={info['p_mante']} slots={info['slots']} soroll_j={info['soroll_jugadors_sigma_m']}", flush=True)


if __name__ == "__main__":
    main()
