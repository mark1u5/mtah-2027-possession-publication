"""Agregats del candidat v7 al partit proporcionat (part 2). Només imprimeix agregats.

    $P bench/partit_v7.py [--out bench/results/partit_v7.json]

Models: v7 (porta amb la TCN v7b i les regles p3), porta_v7b_vora (el mateix amb janv1_notag),
janv1 p3 (la reserva) i janv1_vit. Per a cada un: possessions, durada mitjana, % de D i % de X,
comprovació de domini, % de frames dubtosos de la porta i, entre v7 i porta_v7b_vora, % de frames
amb etiqueta diferent. La columna pública és informativa i va a part.
"""
import argparse
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
ARREL = os.path.dirname(AQUI)
sys.path.insert(0, ARREL)
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(ARREL, "evaluation"))
import adapter  # noqa: E402
import domini  # noqa: E402
import harness  # noqa: E402
import possession_metrics as pm  # noqa: E402
import tune_decoder as td  # noqa: E402
from models import v7  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(AQUI, "results", "partit_v7.json"))
    a = ap.parse_args()
    d = adapter.carrega_partit()
    graded = np.zeros(d["n_frames"], bool)
    for _, x, y in d["half_blocks"]:
        graded[x:y + 1] = True

    et, dubtos = {}, {}
    et["v7"], clar = v7.prediu(d)
    dubtos["v7"] = 100.0 * float(np.mean(~clar[graded]))
    et["porta_v7b_vora"] = harness.MODELS["porta_v7b_vora"][0](d)
    dubtos["porta_v7b_vora"] = 100.0 * float(np.mean(~td.janv1_amb_control(d)[1][graded]))
    et["janv1"] = harness.MODELS["janv1"][0](d)
    et["janv1_vit"] = harness.MODELS["janv1_vit"][0](d)

    from run_evaluation import find_gt
    gt = pm.label_codes(pm.read_label_csv(find_gt("public")))
    pr = domini.prior()
    out = {"_prior_domini": pr, "models": {}}
    for m, lab in et.items():
        s = harness.sanitat_partit(lab, d)
        s["domini"] = domini.dins(s["possessions"], pr)
        s["pct_dubtosos_porta"] = round(dubtos[m], 2) if m in dubtos else None
        r = pm.evaluate(gt, pm.label_codes(lab))
        out["models"][m] = {"sanitat": s, "public_no_s_hi_decideix": {"spellF1": round(r["spell_f1_mean"], 4),
                                                                      "macroF1": round(r["macro_f1"], 4)}}
    dif = np.asarray(et["v7"])[graded] != np.asarray(et["porta_v7b_vora"])[graded]
    out["pct_frames_diferents_v7_porta_v7b"] = round(100.0 * float(dif.mean()), 2)
    v = out["models"]["v7"]["sanitat"]["domini"]
    p = out["models"]["porta_v7b_vora"]["sanitat"]["domini"]
    out["candidat"] = "v7" if v else ("porta_v7b_vora" if p else None)

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"prior {pr['prior']}, interval {pr['interval']}")
    print("model | possessions | durada s | % D | % X | domini | % dubtosos")
    for m, r in out["models"].items():
        s = r["sanitat"]
        print(f"{m} | {s['possessions']} | {s['durada_mitjana_s']} | {s['pct_D']} | {s['pct_X']} | "
              f"{'dins' if s['domini'] else 'fora'} | {s['pct_dubtosos_porta']}")
    print(f"% frames diferents entre v7 i porta_v7b_vora: {out['pct_frames_diferents_v7_porta_v7b']}")
    print(f"candidat: {out['candidat']}")
    print("públic (informatiu, no s'hi decideix res): " +
          "; ".join(f"{m} {r['public_no_s_hi_decideix']['spellF1']} / {r['public_no_s_hi_decideix']['macroF1']}"
                    for m, r in out["models"].items()))


if __name__ == "__main__":
    main()
