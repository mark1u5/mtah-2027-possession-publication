# Ball Possession from Tracking Data (`v9m1`)

Submission pipeline **v9m1** (`markius_v9_m1`) by **Marc Bisbal & Jan Fusté** (Team El Chato) for *More Than A Hack 2027* (BAR–ATH). For every frame of the tactical video, the model predicts `H` (home in possession), `A` (away in possession), `D` (dead ball), or `X` (outside the halves), reading **only the ATD tracking**.

- **Technical document**: [`TECHNICAL_DOCUMENT.md`](TECHNICAL_DOCUMENT.md) · [`TECHNICAL_DOCUMENT.pdf`](TECHNICAL_DOCUMENT.pdf)
- **Interactive online demo**: [`demo/`](demo/README.md) (2D pitch replay, synchronized `TCN v7d` + `p3_notag` + `conf_fisica` + `Viterbi` tracks, and live `v9m1` parameter lab on open Metrica sample matches)
- **Archived final ensemble (`Nausicaa`)**: preserved on branch `archive/nausicaa`

## Model overview (`v9m1`)

1. **Learned temporal branch**: 32 kinematic features (`base/bench/features.py`) fed to a non-causal dilated TCN (`models/final/tcn.pt`, `tcn_v7d`, ±20 s receptive field).
2. **Physical rules branch**: deterministic rule set `principled_v1` (`p3_notag`, no Smart Tagging) plus invariant kinematic contact confidence `conf_fisica` ($\kappa \le 0.95$, decay $0.20\text{ (m/s)}^{-1}$).
3. **Physically modulated Viterbi decoder**: dynamically lowers TCN weight under clear physical contact ($w_t = w\,(1 - \text{factor}\cdot\text{conf}_t)$ with `w=1.0`, `factor=0.35`, `lam=0.25`) and enforces decoupled minimum spell durations (`s_P=1.0` for `H/A`, `s_D=0.25` for `D`).

On the 10-match external benchmark (`results/v9m1/`), `v9m1` achieves **0.701 Spell-F1** `[0.667, 0.735]` and **0.893 Macro-F1** (`0.714` on IDSSE, `0.671` on hold-out Metrica).

## Reproduce the submission file

Python 3.12, CPU only. The competition match data is not in this repository (not redistributable).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python predict.py --data <data_folder> --out submission.csv
```

- `<data_folder>` must contain `tactical/BAR-ATH_tactical_atd.txt` and `.xml` (or set `MTAH_DATA_DIR`, or place them under `./data`). No label or Smart Tagging file is read.
- Expected output SHA-256 (`submission.csv` and `submission.csv.sha256`):
  `b9cab7198add0f670849b0c34e92702d50c9a5703b8231a36a96a1cc287825ed`
- Validate format:
  `python evaluation/validate_submission.py submission.csv`

## Run tests & interactive demo

```bash
# Unit & synthetic end-to-end tests (no match data required)
python tests/test_decoder.py
python tests/test_v9m1_inference.py

# Interactive demo (works statically or via local Python server)
python demo/server.py --port 4174
```

## Contents

| Path | Description |
|---|---|
| `predict.py` | Single-command deterministic inference for `v9m1` |
| `models/final/` | `tcn.pt` (`tcn_v7d_weights.pt`) and `params.json` (`v9m1` decoder settings + HMM stats) |
| `base/` | Shared inference modules: ATD adapter, features, TCN (`tcn_v7.py`), rules (`principled_v1.py`), Viterbi decoder (`decoder.py`) |
| `demo/` | Interactive web demo (`index.html`, JS Viterbi + `spellF1` engine, 2D pitch + live parameter lab, Docker/Caddy deploy) |
| `dataset/` | External match degradation (`v3`) and label-free calibration checks |
| `results/v9m1/` | Benchmark results on external matches (JSON + Markdown reports) |
| `fig/` | Vector SVG figures, benchmark data, and PDF builder (`fig/figs_svg.py`, `fig/build_pdf.py`) |
| `tests/` | Synthetic unit & end-to-end inference tests |
| `requirements.lock` | Pinned CPU inference dependencies |

## Licence

MIT (see [`LICENSE`](LICENSE)). `base/games.py` originates from the challenge starter repository (MIT, © Metrica Sports). Demo sample windows use open anonymised tracking from [Metrica Sports Sample Data](https://github.com/metrica-sports/sample-data).
