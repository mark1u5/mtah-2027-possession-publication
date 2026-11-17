"""Estadístiques agregades del tracking del partit proporcionat (objectiu de la degradació).

Només fa servir el tracking automàtic (ATD tactical), mai la GT. Compten els frames de les dues
parts segons games.half_video_blocks; no es condiciona a cap estat de pilota ni possessió.
Desa bench/target_stats.json i imprimeix només agregats.

    /workspace/shared/venvs/bench/bin/python bench/match_stats.py
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
ARREL = os.path.dirname(AQUI)
sys.path.insert(0, ARREL)
sys.path.insert(0, AQUI)
import games  # noqa: E402
import tracking_stats as ts  # noqa: E402


def equips_slots(meta_path):
    # mateix criteri que starter/principled_v1.py: el primer <Team> és el local
    root = ET.parse(meta_path).getroot()
    ids = [t.get("id") for t in root.iter("Team")]
    return np.array([ids.index(p.get("teamId")) if p.get("teamId") in ids else -1
                     for p in root.iter("Player")], np.int8)


def llegeix_atd(raw_path, n, n_slots, L, W):
    """Posicions en metres (origen al centre). Format: frame:slot;slot;...:pilota, x i y de 0 a 1."""
    xy = np.full((n, n_slots, 2), np.nan, np.float32)
    bola = np.full((n, 2), np.nan, np.float32)
    with open(raw_path) as fh:
        for linia in fh:
            parts = linia.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1  # frame de vídeo
            if not 0 <= f < n:
                continue
            for s, g in enumerate(parts[1].split(";")):
                if g and g[0] != "N":
                    c = g.split(",")
                    xy[f, s] = ((float(c[0]) - 0.5) * L, (0.5 - float(c[1])) * W)
            b = parts[2]
            if b and b[0] != "N":
                c = b.split(",")
                bola[f] = ((float(c[0]) - 0.5) * L, (0.5 - float(c[1])) * W)
    return bola, xy


def main():
    p = games.asset_paths()
    L, W = games.GAMES[games.GAME]["pitch"]
    n = games.source_frames()
    equip = equips_slots(p["atd_meta"])
    bola, xy = llegeix_atd(p["atd_raw"], n, len(equip), L, W)
    mask = np.zeros(n, bool)
    for _, a, b in games.half_video_blocks(games.GAME):
        mask[a:b + 1] = True
    s = ts.estadistiques(bola, xy, equip, mask, bola_z=None, camp=(L, W))
    s["origen"] = {"font": "ATD tactical del partit proporcionat (tracking automàtic, sense GT)",
                   "frames": "les dues parts segons games.half_video_blocks",
                   "camp_m": [L, W], "alcada_pilota": False}
    with open(os.path.join(AQUI, "target_stats.json"), "w") as fh:
        json.dump(s, fh, indent=1, ensure_ascii=False, sort_keys=True)
    print(json.dumps(ts.resum(s), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
