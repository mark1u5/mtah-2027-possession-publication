# Interactive `v9m1` Online Demo

An interactive browser-based explorer for **v9m1** (`markius_v9_m1`):

- **2D Tactical Pitch** with smooth player interpolation, ball trail, kinematic contact confidence ring ($\kappa \le 0.95$), and live ATD ball-dropout indicators.
- **Synchronized 4-Track Timeline**: `TCN v7d` probabilities, `p3_notag` rules + `conf_fisica`, `v9m1` Viterbi spells, and Ground Truth.
- **Live Viterbi Parameter Lab**: adjust `w`, `factor`, `lam`, `s_P`, and `s_D` in real time (~5 ms in-browser exact Viterbi + `Spell-F1` / `Macro-F1` scoring).
- **Open Data Only**: uses 4-minute windows from [Metrica Sports Sample Data](https://github.com/metrica-sports/sample-data) (`Sample_Game_1` clean & `v3`-degraded, `Sample_Game_2` `v3`-degraded). No restricted competition match data is exposed.

## Run locally

```bash
python demo/server.py --port 4174
# Open http://localhost:4174
```

## Deploy online (2 zero-friction options)

### Option 1: GitHub Pages (free, static)

Because Viterbi and `Spell-F1` run 100% client-side in `demo/viterbi.js`, no backend is required:

1. Push `main` to GitHub.
2. In repository **Settings → Pages**, set **Deploy from a branch** → `main` / `/ (root)`.
3. Open `https://<username>.github.io/mtah-2027-possession-publication/demo/`.

### Option 2: VPS + `sslip.io` HTTPS (like `64-177-60-166.sslip.io`)

On any Linux VPS with Docker installed (replace `YOUR.VPS.IP` with dashes, e.g. `64-177-60-166.sslip.io`):

```bash
cd demo
DEMO_HOST=YOUR-VPS-IP.sslip.io docker compose up -d --build
```

Caddy automatically provisions a Let's Encrypt TLS certificate for `https://YOUR-VPS-IP.sslip.io`.
