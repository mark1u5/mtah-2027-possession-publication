"""Build TECHNICAL_DOCUMENT.pdf from plain Python (no external PDF/HTML packages required)."""
import os
import sys
import zlib


def _pdf_esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap(text, max_chars=95):
    words = text.split()
    lines, cur = [], []
    n = 0
    for w in words:
        if cur and n + 1 + len(w) > max_chars:
            lines.append(" ".join(cur))
            cur, n = [w], len(w)
        else:
            cur.append(w)
            n += (1 if cur[:-1] else 0) + len(w)
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


class PDFDoc:
    def __init__(self):
        self.pages = []
        self.cur = []
        self.y = 785
        self.w = 595
        self.h = 842
        self.margin = 48

    def new_page(self):
        if self.cur:
            self.pages.append("\n".join(self.cur))
        self.cur = []
        self.y = 792

    def ensure(self, pts):
        if self.y - pts < 52:
            self.new_page()

    def text(self, x, y, s, font="F1", size=9.5, rgb=(0.08, 0.11, 0.16)):
        r, g, b = rgb
        self.cur.append(
            f"BT /{font} {size:.1f} Tf {r:.3f} {g:.3f} {b:.3f} rg "
            f"1 0 0 1 {x:.1f} {y:.1f} Tm ({_pdf_esc(s)}) Tj ET"
        )

    def rect(self, x, y, w, h, rgb=(0.92, 0.94, 0.97)):
        r, g, b = rgb
        self.cur.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x:.1f} {y:.1f} {w:.1f} {h:.1f} re f")

    def line(self, x0, y0, x1, y1, rgb=(0.75, 0.80, 0.86), lw=0.6):
        r, g, b = rgb
        self.cur.append(f"{r:.3f} {g:.3f} {b:.3f} RG {lw:.2f} w {x0:.1f} {y0:.1f} m {x1:.1f} {y1:.1f} l S")

    def heading1(self, title, subtitle):
        self.ensure(46)
        self.text(self.margin, self.y, title, font="F2", size=17, rgb=(0.04, 0.18, 0.35))
        self.y -= 18
        self.text(self.margin, self.y, subtitle, font="F1", size=9.5, rgb=(0.35, 0.40, 0.48))
        self.y -= 12
        self.line(self.margin, self.y, self.w - self.margin, self.y, rgb=(0.11, 0.54, 0.56), lw=1.2)
        self.y -= 16

    def heading2(self, title):
        self.ensure(30)
        self.y -= 6
        self.text(self.margin, self.y, title, font="F2", size=11.5, rgb=(0.04, 0.18, 0.35))
        self.y -= 5
        self.line(self.margin, self.y, self.w - self.margin, self.y, rgb=(0.78, 0.82, 0.87), lw=0.5)
        self.y -= 13

    def para(self, text, size=9.3):
        lines = _wrap(text, 96)
        self.ensure(len(lines) * 13 + 6)
        for ln in lines:
            self.text(self.margin, self.y, ln, font="F1", size=size)
            self.y -= 12.8
        self.y -= 3

    def bullet(self, lead, body):
        full = f"- {lead}: {body}" if lead else f"- {body}"
        lines = _wrap(full, 94)
        self.ensure(len(lines) * 12.5 + 4)
        for i, ln in enumerate(lines):
            self.text(self.margin + (8 if i else 0), self.y, ln, font="F1", size=9.2)
            self.y -= 12.5
        self.y -= 1.5

    def code_block(self, lines):
        bh = len(lines) * 12 + 10
        self.ensure(bh + 8)
        self.rect(self.margin, self.y - bh + 8, self.w - 2 * self.margin, bh, rgb=(0.95, 0.96, 0.98))
        yy = self.y - 4
        for ln in lines:
            self.text(self.margin + 8, yy, ln, font="F3", size=8.5, rgb=(0.10, 0.14, 0.22))
            yy -= 12
        self.y -= bh + 6

    def table(self, headers, rows, widths):
        total_w = self.w - 2 * self.margin
        col_w = [total_w * w / 100.0 for w in widths]
        # Header
        self.ensure(28)
        self.rect(self.margin, self.y - 14, total_w, 17, rgb=(0.90, 0.93, 0.97))
        cx = self.margin + 4
        for h, cw in zip(headers, col_w):
            self.text(cx, self.y - 9, h, font="F2", size=8.4, rgb=(0.04, 0.18, 0.35))
            cx += cw
        self.y -= 19
        for row in rows:
            wrapped = [_wrap(str(cell), max(12, int(cw / 4.7))) for cell, cw in zip(row, col_w)]
            max_lines = max(len(w) for w in wrapped)
            rh = max_lines * 11 + 6
            self.ensure(rh + 4)
            cx = self.margin + 4
            for wlines, cw in zip(wrapped, col_w):
                for li, ln in enumerate(wlines):
                    self.text(cx, self.y - 8 - li * 11, ln, font="F1", size=8.2)
                cx += cw
            self.line(self.margin, self.y - rh + 2, self.margin + total_w, self.y - rh + 2, rgb=(0.88, 0.90, 0.93), lw=0.4)
            self.y -= rh

    def save(self, path):
        if self.cur:
            self.pages.append("\n".join(self.cur))
            self.cur = []
        objs = []
        # 1: Catalog, 2: Pages, 3: F1 Helvetica, 4: F2 Helvetica-Bold, 5: F3 Courier
        page_ids = []
        for idx in range(len(self.pages)):
            page_ids.append(6 + idx * 2)
        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        objs.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
        objs.append(f"2 0 obj << /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >> endobj\n".encode("ascii"))
        objs.append(b"3 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
        objs.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> endobj\n")
        objs.append(b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Courier >> endobj\n")
        for idx, content in enumerate(self.pages):
            pid = 6 + idx * 2
            cid = pid + 1
            raw = content.encode("latin-1", errors="replace")
            comp = zlib.compress(raw)
            objs.append(
                f"{pid} 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.w} {self.h}] "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> /Contents {cid} 0 R >> endobj\n".encode("ascii")
            )
            objs.append(
                f"{cid} 0 obj << /Length {len(comp)} /Filter /FlateDecode >>\nstream\n".encode("ascii")
                + comp
                + b"\nendstream\nendobj\n"
            )
        buf = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for obj in objs:
            offsets.append(len(buf))
            buf.extend(obj)
        xref_pos = len(buf)
        buf.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("ascii"))
        for off in offsets[1:]:
            buf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
        buf.extend(f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii"))
        with open(path, "wb") as fh:
            fh.write(buf)


def build(dst="TECHNICAL_DOCUMENT.pdf"):
    doc = PDFDoc()
    doc.heading1(
        "Ball Possession from ATD Tracking: v9m1",
        "Team El Chato · More Than A Hack 2027 (BAR-ATH) · Input: tactical ATD tracking only",
    )
    doc.table(
        ["Headline metric", "External benchmark", "Input", "Reproducibility"],
        [[
            "0.701 Spell-F1 [0.667, 0.735]\nMacro-F1: 0.893",
            "10 external matches (7 IDSSE + 3 Metrica)\n+0.0047 over markius_v9 (6/10 wins)",
            "ATD tracking only\nNo video, events, or ST",
            "Byte-exact CSV\nsha256: b9cab719...25ed",
        ]],
        [26, 30, 22, 22],
    )

    doc.heading2("1. Approach (v9m1 Architecture)")
    doc.para(
        "v9m1 combines two independent branches inside a physically modulated hidden Markov model "
        "decoded with Viterbi: (1) a non-causal Temporal Convolutional Network (TCN v7d, 8 dilated "
        "residual blocks, receptive field +/-20 s) trained on degraded external matches, and (2) a "
        "deterministic physical rule set (principled_v1 p3_notag, without Smart Tagging) paired with "
        "an invariant kinematic contact confidence conf_fisica."
    )
    doc.table(
        ["Stage", "Role", "Mechanism"],
        [
            ["1. Features", "32 kinematic inputs", "Ball coords/speed/accel, gaps, pitch boundaries, closest 2 players per team, centroids"],
            ["2. TCN (v7d)", "Per-frame H/A/D probs", "Dilations 1..128 (+/-510 frames), triple loss weight near transitions, v3 degradation"],
            ["3. Rules & Conf", "p3_notag + conf_fisica", "Closest reachable player + relative speed control; kappa <= 0.95, exp(-0.20 * rel_speed)"],
            ["4. Modulation", "Dynamic log-linear", "w_t = w * (1 - factor * conf_t) with w=1.0, factor=0.35, lam=0.25"],
            ["5. Viterbi HMM", "Decoupled min durations", "s_dur = [1.0, 1.0, 0.25] (s_P=1.0 for live H/A, s_D=0.25 for dead ball D)"],
        ],
        [18, 24, 58],
    )

    doc.heading2("2. Possession Rules & Physical Constants")
    doc.bullet("Live vs dead ball", "Ball out, stoppage, or visibly stationary play until first restart touch -> D; controlled action -> H/A from first touch; loose balls/deflections keep prior possessor.")
    doc.bullet("No Smart Tagging", "Uses p3_notag (ATD tracking only). Three candidate rule variants (contest, control, restart) were screened on external matches and all rejected (-0.0002 to -0.0009 spell-F1).")
    doc.bullet("Decoupled durations", "Allowing shorter dead-ball segments (s_D=0.25 vs s_P=1.0) recovers brief stoppages without over-fragmenting live H/A spells.")

    doc.heading2("3. Training Data & External Benchmark")
    doc.table(
        ["Dataset", "Matches", "Role", "Notes"],
        [
            ["IDSSE (Bundesliga)", "7 matches (~11 h)", "TCN + decoder tuning", "DFL / Sportec optical tracking, degraded with v3"],
            ["SkillCorner Open", "20 A-League matches", "TCN training + stats", "Automatic broadcast tracking (informative eval only)"],
            ["Metrica Sample Data", "3 open matches", "Hold-out test only", "Never seen by TCN or decoder tuning"],
        ],
        [24, 20, 22, 34],
    )

    doc.heading2("4. Results & Ablation Progression")
    doc.table(
        ["Model", "Spell-F1 (10) [95% CI]", "Macro-F1", "IDSSE / Metrica", "BAR-ATH sanity"],
        [
            ["tcn_v7d (raw argmax)", "0.320 [0.296, 0.343]", "0.891", "0.321 / 0.319", "674 spells, 5.57 s"],
            ["janv1_notag (rules)", "0.478 [0.436, 0.526]", "0.761", "0.434 / 0.582", "345 spells, 9.98 s"],
            ["janv1_vit (rules+HMM)", "0.583 [0.562, 0.606]", "0.840", "0.577 / 0.596", "305 spells, 11.25 s"],
            ["porta_v7d", "0.690 [0.654, 0.723]", "0.891", "0.701 / 0.662", "196 spells, 17.76 s"],
            ["fusio_v7d", "0.696 [0.664, 0.728]", "0.891", "0.705 / 0.676", "208 spells, 17.84 s"],
            ["markius_v9", "0.697 [0.664, 0.727]", "0.892", "0.708 / 0.670", "208 spells, 18.31 s"],
            ["v9m1 (markius_v9_m1)", "0.701 [0.667, 0.735]", "0.893", "0.714 / 0.671", "211 spells, 18.06 s, 40.0% D"],
        ],
        [24, 23, 12, 19, 22],
    )

    doc.heading2("5. Diagnosis, Assets Dropped & Limitations")
    doc.bullet("Segmentation diagnosis", "v9m1 predicts 0.87x true possessions on IDSSE and 0.81x on Metrica (IoU 0.50 precision/recall: 0.78/0.68 IDSSE, 0.74/0.64 Metrica); ~47-51% of H<->A switches are missed.")
    doc.bullet("Real-tracker dead-ball delay", "On Metrica, live->dead transitions arrive +6.4 frames late on average (83% late) when the tracker drops the ball leaving play and stoppages are inferred from player deceleration.")
    doc.bullet("Rejected variants", "v9_r (neutral TCN in gaps: 0.688), v9m1x extended grid (0.701 on label_acc but worse on label_dur: 0.687 vs 0.693), symmetric H/A HMM (-0.0003), and rule variants contest/control/restart.")

    doc.heading2("6. Reproduction")
    doc.code_block([
        "python3.12 -m venv .venv",
        ".venv/bin/pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu",
        ".venv/bin/python predict.py --data <data_folder> --out submission.csv",
    ])
    doc.para("Expected output sha256 (CPU, 2 threads, deterministic): b9cab7198add0f670849b0c34e92702d50c9a5703b8231a36a96a1cc287825ed.")
    doc.save(dst)
    print("Wrote", dst)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "TECHNICAL_DOCUMENT.pdf")
