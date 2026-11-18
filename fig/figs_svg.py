"""Generate crisp SVG figures for the v9m1 technical document (zero external dependencies).

    python fig/figs_svg.py fig fig/fig_data.json
"""
import html
import json
import os
import sys

NAVY = "#0b2e59"
TEAL = "#1b8a8f"
ORANGE = "#e07b24"
GREY = "#7c8798"
LIGHT = "#eef3f9"
WARM = "#f8efe4"
TEXT = "#18202c"
MUTED = "#596579"


def _esc(s):
    return html.escape(str(s), quote=True)


def _svg_open(w, h):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" font-family="DejaVu Sans, Arial, sans-serif">'
        f'<rect width="{w}" height="{h}" fill="#ffffff" rx="8"/>'
    )


def fig1_pipeline(out_dir):
    w, h = 980, 320
    p = [_svg_open(w, h)]
    p.append(
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="8.5" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="{GREY}"/></marker></defs>'
    )
    # Branch bands
    p.append(f'<rect x="155" y="18" width="380" height="128" rx="12" fill="{LIGHT}"/>')
    p.append(f'<text x="170" y="39" font-size="11" font-weight="700" fill="{NAVY}">LEARNED BRANCH</text>')
    p.append(f'<rect x="155" y="168" width="380" height="132" rx="12" fill="{WARM}"/>')
    p.append(f'<text x="170" y="189" font-size="11" font-weight="700" fill="#8a4f16">PHYSICS &amp; RULES BRANCH</text>')

    def box(x, y, bw, bh, title, sub, bg):
        p.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="10" fill="{bg}"/>')
        p.append(
            f'<text x="{x + bw/2}" y="{y + bh*0.42}" text-anchor="middle" '
            f'font-size="13" font-weight="700" fill="#ffffff">{_esc(title)}</text>'
        )
        lines = sub.split("\n")
        for idx, ln in enumerate(lines):
            p.append(
                f'<text x="{x + bw/2}" y="{y + bh*0.67 + idx*14}" text-anchor="middle" '
                f'font-size="10.5" fill="#e6edf5">{_esc(ln)}</text>'
            )

    def arrow(x0, y0, x1, y1, label=None, lx=0, ly=0):
        p.append(
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" '
            f'stroke="{GREY}" stroke-width="2" marker-end="url(#arr)"/>'
        )
        if label:
            for idx, ln in enumerate(label.split("\n")):
                p.append(
                    f'<text x="{lx}" y="{ly + idx*12}" text-anchor="middle" '
                    f'font-size="10" font-style="italic" fill="{MUTED}">{_esc(ln)}</text>'
                )

    box(16, 114, 110, 88, "ATD", "Ball + 22 players\n25 fps", GREY)
    box(174, 52, 150, 78, "1 · Features", "32/35 kinematic\nfeatures per frame", NAVY)
    box(360, 52, 156, 78, "2 · TCN (v7d)", "8 dilated blocks\n±20 s context", NAVY)
    box(174, 202, 158, 80, "3a · Rules", "principled_v1\n(p3_notag, no ST)", NAVY)
    box(360, 202, 156, 80, "3b · Contact conf", "κ ≤ 0.95 decay\nconf_fisica(t)", NAVY)
    box(568, 108, 178, 98, "4 · Modulation", "w_t = w(1 − α·conf_t)\nlog-linear emissions", TEAL)
    box(778, 108, 112, 98, "5 · Viterbi", "Decoupled min\nduration (s_P, s_D)", NAVY)
    box(914, 108, 52, 98, "CSV", "H/A/D\n/ X", ORANGE)

    arrow(126, 144, 170, 96)
    arrow(126, 174, 170, 240)
    arrow(324, 91, 356, 91)
    arrow(332, 242, 356, 242)
    arrow(516, 91, 564, 138, "p_tcn(H, A, D)", 558, 78)
    arrow(516, 232, 564, 184, "q_rules + w_t", 554, 250)
    arrow(746, 157, 774, 157)
    arrow(890, 157, 910, 157)
    p.append("</svg>")
    with open(os.path.join(out_dir, "fig1_pipeline.svg"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(p))


def fig2_results(out_dir, data):
    w, h = 980, 320
    p = [_svg_open(w, h)]
    matches = data["matches"]
    fusio = data["fusio_v7d"]
    v9m1 = data["v9m1"]

    x0, y0, pw, ph = 64, 34, 892, 206
    y_min, y_max = 0.50, 0.85
    p.append(f'<text x="{w/2}" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="{TEXT}">Per-match Spell-F1 on the 10 external benchmark matches</text>')

    for tick in (0.50, 0.60, 0.70, 0.80):
        yy = y0 + ph - (tick - y_min) / (y_max - y_min) * ph
        p.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="#e5e9f0" stroke-width="1"/>')
        p.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" font-size="10.5" fill="{MUTED}">{tick:.2f}</text>')

    sep_x = x0 + 7 * (pw / len(matches))
    p.append(f'<line x1="{sep_x:.1f}" y1="{y0}" x2="{sep_x:.1f}" y2="{y0+ph}" stroke="#b8c2d0" stroke-dasharray="4 4"/>')
    p.append(f'<text x="{x0 + 3.5*(pw/len(matches))}" y="{y0+14}" text-anchor="middle" font-size="10.5" fill="{MUTED}">IDSSE (Bundesliga · degraded v3)</text>')
    p.append(f'<text x="{x0 + 8.5*(pw/len(matches))}" y="{y0+14}" text-anchor="middle" font-size="10.5" fill="{MUTED}">Metrica (never trained on)</text>')

    slot = pw / len(matches)
    bw = slot * 0.34
    for i, m in enumerate(matches):
        cx = x0 + (i + 0.5) * slot
        for val, dx, col in ((fusio[i], -bw - 2, GREY), (v9m1[i], 2, TEAL)):
            bh = max(0.0, (val - y_min) / (y_max - y_min) * ph)
            yy = y0 + ph - bh
            p.append(f'<rect x="{cx+dx:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{col}"/>')
            p.append(f'<text x="{cx+dx+bw/2:.1f}" y="{yy-4:.1f}" text-anchor="middle" font-size="9" fill="{TEXT}">{val:.3f}</text>')
        short = m.split("/")[1].replace("DFL-MAT-", "").replace("Sample_Game_", "Metrica ")
        p.append(f'<text x="{cx:.1f}" y="{y0+ph+16}" text-anchor="middle" font-size="10" fill="{TEXT}">{_esc(short)}</text>')

    p.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="#94a3b8"/>')
    # Legend
    mean_f = sum(fusio) / len(fusio)
    mean_v = sum(v9m1) / len(v9m1)
    p.append(f'<rect x="220" y="282" width="12" height="12" rx="2" fill="{GREY}"/>')
    p.append(f'<text x="238" y="292" font-size="11" fill="{TEXT}">fusio_v7d (fixed fusion · mean {mean_f:.3f})</text>')
    p.append(f'<rect x="540" y="282" width="12" height="12" rx="2" fill="{TEAL}"/>')
    p.append(f'<text x="558" y="292" font-size="11" font-weight="700" fill="{TEXT}">v9m1 (contact modulation + decoupled Viterbi · mean {mean_v:.3f})</text>')
    p.append("</svg>")
    with open(os.path.join(out_dir, "fig2_results.svg"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(p))


def fig3_progression(out_dir, data):
    w, h = 980, 290
    p = [_svg_open(w, h)]
    stages = [
        ("tcn_v7d", "Raw TCN\n(no decoder)", GREY),
        ("janv1_notag", "Rules only\n(p3_notag)", GREY),
        ("janv1_vit", "Rules +\nViterbi", GREY),
        ("porta_v7d", "Gate +\nViterbi", NAVY),
        ("fusio_v7d", "Fixed fusion\n+ Viterbi", NAVY),
        ("markius_v9", "+ Contact\nmodulation", NAVY),
        ("markius_v9_m1", "v9m1 (+ decoupled\nmin durations)", TEAL),
    ]
    x0, y0, pw, ph = 70, 36, 875, 178
    y_min, y_max = 0.25, 0.78
    p.append(f'<text x="{w/2}" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="{TEXT}">Progression across stages (Spell-F1 on 10 external matches, 95% bootstrap CI)</text>')

    for tick in (0.30, 0.40, 0.50, 0.60, 0.70):
        yy = y0 + ph - (tick - y_min) / (y_max - y_min) * ph
        p.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="#e5e9f0"/>')
        p.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" font-size="10.5" fill="{MUTED}">{tick:.2f}</text>')

    pts = []
    slot = pw / len(stages)
    for i, (key, label, col) in enumerate(stages):
        mean, lo, hi = data["summary"][key]
        cx = x0 + (i + 0.5) * slot
        ym = y0 + ph - (mean - y_min) / (y_max - y_min) * ph
        ylo = y0 + ph - (lo - y_min) / (y_max - y_min) * ph
        yhi = y0 + ph - (hi - y_min) / (y_max - y_min) * ph
        pts.append((cx, ym, ylo, yhi, mean, label, col))

    for i in range(len(pts) - 1):
        p.append(f'<line x1="{pts[i][0]:.1f}" y1="{pts[i][1]:.1f}" x2="{pts[i+1][0]:.1f}" y2="{pts[i+1][1]:.1f}" stroke="#cbd5e1" stroke-width="2.2"/>')

    for cx, ym, ylo, yhi, mean, label, col in pts:
        p.append(f'<line x1="{cx:.1f}" y1="{ylo:.1f}" x2="{cx:.1f}" y2="{yhi:.1f}" stroke="{col}" stroke-width="2" opacity="0.55"/>')
        p.append(f'<circle cx="{cx:.1f}" cy="{ym:.1f}" r="6.5" fill="{col}"/>')
        p.append(f'<text x="{cx:.1f}" y="{yhi - 8:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="{col}">{mean:.3f}</text>')
        for j, ln in enumerate(label.split("\n")):
            p.append(f'<text x="{cx:.1f}" y="{y0 + ph + 18 + j*13}" text-anchor="middle" font-size="10" fill="{TEXT}">{_esc(ln)}</text>')

    p.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="#94a3b8"/>')
    p.append("</svg>")
    with open(os.path.join(out_dir, "fig3_progression.svg"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(p))


def fig4_diagnosis(out_dir):
    w, h = 980, 265
    p = [_svg_open(w, h)]
    # Left panel: Predicted / Real possessions ratio & IoU=0.5 Precision vs Recall
    p.append(f'<text x="245" y="24" text-anchor="middle" font-size="12.5" font-weight="700" fill="{TEXT}">Segmentation diagnosis (IoU 0.50)</text>')
    datasets = [("IDSSE (7)", 0.78, 0.68, 0.87), ("Metrica (3)", 0.74, 0.64, 0.81)]
    lx0, ly0, lw, lh = 65, 48, 370, 155
    for tick in (0.0, 0.25, 0.50, 0.75, 1.0):
        yy = ly0 + lh - tick * lh
        p.append(f'<line x1="{lx0}" y1="{yy:.1f}" x2="{lx0+lw}" y2="{yy:.1f}" stroke="#e5e9f0"/>')
        p.append(f'<text x="{lx0-8}" y="{yy+4:.1f}" text-anchor="end" font-size="10" fill="{MUTED}">{tick:.2f}</text>')
    for i, (name, prec, rec, ratio) in enumerate(datasets):
        gx = lx0 + 42 + i * 180
        for val, dx, col in ((prec, 0, NAVY), (rec, 36, TEAL), (ratio, 72, ORANGE)):
            bh = val * lh
            yy = ly0 + lh - bh
            p.append(f'<rect x="{gx+dx}" y="{yy:.1f}" width="30" height="{bh:.1f}" rx="3" fill="{col}"/>')
            p.append(f'<text x="{gx+dx+15}" y="{yy-5:.1f}" text-anchor="middle" font-size="9.5" font-weight="700" fill="{TEXT}">{val:.2f}</text>')
        p.append(f'<text x="{gx+51}" y="{ly0+lh+16}" text-anchor="middle" font-size="10.5" fill="{TEXT}">{name}</text>')
    p.append(f'<line x1="{lx0}" y1="{ly0+lh}" x2="{lx0+lw}" y2="{ly0+lh}" stroke="#94a3b8"/>')
    for x, col, lbl in ((70, NAVY, "Precision"), (165, TEAL, "Recall"), (255, ORANGE, "Pred / Real spells")):
        p.append(f'<rect x="{x}" y="236" width="10" height="10" rx="2" fill="{col}"/>')
        p.append(f'<text x="{x+15}" y="245" font-size="10" fill="{TEXT}">{lbl}</text>')

    # Right panel: Where errors happen (diagnosis findings)
    rx = 500
    p.append(f'<text x="735" y="24" text-anchor="middle" font-size="12.5" font-weight="700" fill="{TEXT}">Error sources found in external validation</text>')
    items = [
        ("Missed H↔A switches", "47% missed on IDSSE · 51% on Metrica (short turnovers)", 0.51, ORANGE),
        ("Live → Dead delay", "Metrica mean +6.4 frames late (83% late; ball lost out of play)", 0.83, NAVY),
        ("H↔A transition lag", "Metrica p75 +16 frames late (mean +6.8 frames)", 0.66, TEAL),
    ]
    for idx, (title, desc, frac, col) in enumerate(items):
        yy = 56 + idx * 62
        p.append(f'<text x="{rx}" y="{yy}" font-size="11.5" font-weight="700" fill="{TEXT}">{_esc(title)}</text>')
        p.append(f'<text x="{rx}" y="{yy+16}" font-size="10" fill="{MUTED}">{_esc(desc)}</text>')
        p.append(f'<rect x="{rx}" y="{yy+24}" width="440" height="12" rx="6" fill="#edf1f7"/>')
        p.append(f'<rect x="{rx}" y="{yy+24}" width="{440*frac:.1f}" height="12" rx="6" fill="{col}"/>')
    p.append("</svg>")
    with open(os.path.join(out_dir, "fig4_diagnosis.svg"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(p))


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "fig"
    data_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(out_dir, "fig_data.json")
    with open(data_path, encoding="utf-8") as fh:
        data = json.load(fh)
    fig1_pipeline(out_dir)
    fig2_results(out_dir, data)
    fig3_progression(out_dir, data)
    fig4_diagnosis(out_dir)
    print("Generated 4 SVG figures in", out_dir)


if __name__ == "__main__":
    main()
