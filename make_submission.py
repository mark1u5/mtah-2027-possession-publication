"""Genera el CSV de submissió d'un model i el seu sha256, amb una sola comanda.

    python make_submission.py --model janv1 --team TEAM --out $SUBMISSIONS_DIR/
    python make_submission.py --model v7 --team TEAM --out $SUBMISSIONS_DIR/v7/
    python make_submission.py --model v7d --team TEAM --out $SUBMISSIONS_DIR/v7d/
    python make_submission.py --model v9m1 --team TEAM --out $SUBMISSIONS_DIR/v9m1/

$SUBMISSIONS_DIR és el directori compartit de submissions (el valor per defecte de --out).

Escriu <team>.csv i <team>.csv.sha256. Només imprimeix agregats: la sanitat sobre el partit sencer
(possessions, durada mitjana, % de D i % de X) i, amb --public, spellF1 i macroF1 al tram públic,
que són només informatius.

janv1 = starter/principled_v1.py amb la configuració p3. Sempre es recalcula des del tracking, sense
fer servir cap cache, perquè el resultat no depengui de l'estat del directori.

v7 = porta amb la TCN v7b sobre les regles p3 (models/v7.py, docs/v7.md). Carrega
models/tcn_v7b_weights.pt i models/v7_params.json, i corre en CPU amb un nombre fix de fils.

v7d = el mateix muntatge amb la TCN v7d (degradació v3): models/tcn_v7d_weights.pt i
models/v7d_params.json. S'escriu en un directori a part ($SUBMISSIONS_DIR/v7d/).
"""
import argparse
import hashlib
import os
import sys

import numpy as np

ARREL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ARREL)
sys.path.insert(0, os.path.join(ARREL, "evaluation"))
import games  # noqa: E402
import possession_metrics as pm  # noqa: E402


def janv1(game, data_dir):
    from starter import principled_v1 as pv
    cfg = pv.CONFIGS["p3"]
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir)
    n = games.source_frames("tactical", game)
    pitch = games.GAMES[game]["pitch"]
    blocs = games.half_video_blocks(game)
    k2 = games.GAMES[game]["kickoff_2nd"]
    kickoff_2 = (k2["minute"] * 60 + k2["second"]) * pv.FPS + k2["frame"]
    kin = pv.extract_or_load_kinematics(raw_path, meta_path, n, pitch, cache_path=None)
    st = {"home": None, "away": None, "set_pieces": None}
    if cfg["visible_dead"] == "st_veto" or cfg["init"] == "v1":
        st_path = games.asset_paths(game, data_dir)["smart_tagging"]
        st["home"], st["away"], st["set_pieces"] = pv.load_smart_tagging_phases(st_path, n)
    labels, info = pv.predict(kin, st, n, blocs, kickoff_2, pitch, cfg)
    return labels, info


def v7(game, data_dir, fils=2, decimals=None, candidat="v7"):
    # CPU i fils fixos abans d'importar torch
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(fils)
    if data_dir:
        os.environ["MTAH_DATA_DIR"] = data_dir
    sys.path.insert(0, os.path.join(ARREL, "bench"))
    import adapter
    from models import decoder as dc
    from models import v7 as mv7
    d = adapter.carrega_partit()
    tcn, path_params = mv7.CANDIDATS[candidat]
    params = mv7.carrega_params(path_params)
    labels, clar = mv7.regles(d, "p3")
    p = mv7.probs_tcn(d, mv7.pesos(tcn), fils)
    if decimals is not None:
        p = np.round(p, decimals)
    # comprovació de muntatge (bench/check_muntatge.py) abans d'escriure res
    import check_muntatge as cm
    if not cm.imprimeix(cm.comprova_partit(d, p, {"p3": (labels, clar)})):
        sys.exit("error de muntatge: no s'escriu el CSV")
    out =dc.descodifica(p, labels, ~clar, np.asarray(d["period"]), params["st"], [params["cfg"]])[0]
    graded = np.zeros(len(out), bool)
    for _, a, b in d["half_blocks"]:
        graded[a:b + 1] = True
    return out, {"candidat": candidat, "tcn": tcn, "cfg": params["cfg"], "fils": fils, "decimals": decimals,
                 "pct_dubtosos_porta": round(100.0 * float(np.mean(~clar[graded])), 2)}


def v9m1(game, data_dir, fils=2):
    """v9m1: TCN v7d i regles p3_notag (sense Smart Tagging) descodificades amb el v9m1
    (models/markius_v9.py, models/markius_v9m1_params.json). La TCN corre en CPU, determinista i amb
    els mateixos fils que la v7d."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(fils)
    os.environ.setdefault("MTAH_DEG", "v3")
    os.environ.setdefault("MTAH_TCN", "v7d")
    if data_dir:
        os.environ["MTAH_DATA_DIR"] = data_dir
    sys.path.insert(0, os.path.join(ARREL, "bench"))
    import adapter
    import check_muntatge as cm
    import tune_decoder as td
    from models import markius_v9 as mv9
    from models import v7 as mv7
    d = adapter.carrega_partit()
    params = mv9.carrega_params(os.path.join(ARREL, "models", "markius_v9m1_params.json"))
    regles = params.get("regles", "p3_notag")
    p = mv7.probs_tcn(d, mv7.pesos("v7d"), fils)
    d["_p_tcn_v7d"] = p  # el descodificador fa servir aquestes, no en torna a calcular
    labels_j, clar = td.janv1_amb_control(d, regles)
    if not cm.imprimeix(cm.comprova_partit(d, p, {"p3_notag": (labels_j, clar)})):
        sys.exit("error de muntatge: no s'escriu el CSV")
    out = mv9.prediu(d, params)
    return out, {"model": "v9m1", "regles": regles, "cfg": params["cfg"], "fils": fils}


MODELS = {"janv1": janv1, "v7": v7, "v7d": v7, "v9m1": v9m1}


def escriu(labels, path):
    # mateix format que run_pipeline: capçalera i una fila per frame, final de línia \n
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{l}\n" for i, l in enumerate(labels))
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(path + ".sha256", "w") as f:
        f.write(f"{h}  {os.path.basename(path)}\n")
    return h


def sanitat(labels, game):
    codes = pm.label_codes(np.asarray(labels))
    graded = np.zeros(len(codes), bool)
    for _, a, b in games.half_video_blocks(game):
        graded[a:b + 1] = True
    runs = [(a, b + 1) for _, a, b in games.half_video_blocks(game)]
    filled = pm.fill_abstentions(codes, runs)
    st = pm.possession_stats(filled, runs)
    g = codes[graded]
    return {"possessions": st["n_spells"], "durada_mitjana_s": round(st["mean_duration_s"], 2),
            "pct_D": round(100.0 * float(np.mean(g == pm.D)), 2),
            "pct_X": round(100.0 * float(np.mean(g == pm.X)), 2)}


def tram_public(labels):
    from run_evaluation import find_gt
    gt = pm.label_codes(pm.read_label_csv(find_gt("public")))
    r = pm.evaluate(gt, pm.label_codes(np.asarray(labels)))
    return {"spellF1": round(r["spell_f1_mean"], 4), "macroF1": round(r["macro_f1"], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="janv1", choices=sorted(MODELS))
    ap.add_argument("--team", default="TEAM")
    ap.add_argument("--out", default="/workspace/shared/submissions/")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--public", action="store_true", help="spellF1 i macroF1 al tram públic (informatiu)")
    ap.add_argument("--fils", type=int, default=2, help="fils de torch (v7, v7d)")
    ap.add_argument("--decimals", type=int, default=None, help="arrodoneix les probabilitats (v7, v7d)")
    a = ap.parse_args()
    game = games.GAME
    if a.model in ("v7", "v7d"):
        labels, info = v7(game, a.data_dir, a.fils, a.decimals, candidat=a.model)
    elif a.model == "v9m1":
        labels, info = v9m1(game, a.data_dir, a.fils)
    else:
        labels, info = MODELS[a.model](game, a.data_dir)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.team}.csv")
    h = escriu(labels, path)
    if a.model in ("v7", "v7d", "v9m1"):
        print(f"model {a.model}: {info}")
    else:
        print(f"model {a.model}: llindars llegits del tracking {info}")
    print(f"escrit {path}")
    print(f"sha256 {h}")
    print(f"sanitat (partit sencer, frames de les parts): {sanitat(labels, game)}")
    if a.public:
        print(f"tram públic, només informatiu: {tram_public(labels)}")


if __name__ == "__main__":
    main()
