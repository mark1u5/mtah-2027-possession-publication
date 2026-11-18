/**
 * Exact JavaScript port of the v9m1 physically modulated Viterbi decoder (base/models/decoder.py)
 * and the official Spell-F1 / Macro-F1 evaluator (evaluation/possession_metrics.py).
 */

export const CLASSES = ['H', 'A', 'D'];
export const CODE_MAP = { H: 0, A: 1, D: 2, X: 3 };
const EPS_Q = 0.1;
const P_MIN = 1e-6;
const SPELL_IOUS = [0.3, 0.4, 0.5, 0.6, 0.7];

export function parametresHmm(st, sP, sD) {
  const s = [sP, sP, sD];
  const m = [0, 1, 2].map((c) => Math.max(1, Math.round(s[c] * st.p5[c])));
  const pSurt = [0, 1, 2].map((c) => 1.0 / Math.max(1.0, st.mitjana[c] - m[c]));
  const logQueda = pSurt.map((p) => Math.log(1.0 - p));
  const logSurt = [0, 1, 2].map((c) => {
    const row = st.transicions[c];
    const rowSum = row[0] + row[1] + row[2];
    return [0, 1, 2].map((c2) => {
      if (c === c2) return -Infinity;
      const prob = (pSurt[c] * row[c2]) / rowSum;
      return prob > 0 ? Math.log(prob) : -Infinity;
    });
  });
  return { m, logQueda, logSurt };
}

export function decodeV9m1(scenario, cfg) {
  const { signals, st } = scenario;
  const T = scenario.nFrames;
  const { w, factor, lam, s_P, s_D } = cfg;
  const { m, logQueda, logSurt } = parametresHmm(st, s_P, s_D);
  const logPrior = st.prior.map((p) => Math.log(p));

  const logOneMinusEps = Math.log(1.0 - EPS_Q);
  const logHalfEps = Math.log(EPS_Q / 2.0);
  const logUniform = Math.log(1.0 / 3.0);

  // Prefix sums of emissions C[0..(T+1)*3-1]
  const C = new Float64Array((T + 1) * 3);
  const em = new Float64Array(T * 3);
  const wtArr = new Float32Array(T);

  const pH = signals.pH;
  const pA = signals.pA;
  const pD = signals.pD;
  const rules = signals.rules;
  const conf = signals.conf;

  for (let t = 0; t < T; t++) {
    const wt = w * (1.0 - factor * (conf[t] * 0.001));
    wtArr[t] = wt;
    const oneMinusW = 1.0 - wt;

    const rCh = rules[t];
    const rCode = rCh === 'H' ? 0 : rCh === 'A' ? 1 : rCh === 'D' ? 2 : 3;

    const probs = [
      Math.max(pH[t] * 0.001, P_MIN),
      Math.max(pA[t] * 0.001, P_MIN),
      Math.max(pD[t] * 0.001, P_MIN),
    ];

    const base = t * 3;
    const nextBase = (t + 1) * 3;
    for (let c = 0; c < 3; c++) {
      const lq = rCode === 3 ? logUniform : (rCode === c ? logOneMinusEps : logHalfEps);
      const lp = Math.log(probs[c]) - logPrior[c];
      const val = lam * (wt * lp + oneMinusW * lq);
      em[base + c] = val;
      C[nextBase + c] = C[base + c] + val;
    }
  }

  // Single-config Viterbi pass matching base/models/decoder.py
  const ent = new Float64Array(T * 3);
  const arg = new Int8Array(T * 3);
  const L = new Float64Array(T * 3);
  const entrades = [[], [], []];

  ent.fill(-Infinity);
  L.fill(-Infinity);
  ent[0] = logPrior[0];
  ent[1] = logPrior[1];
  ent[2] = logPrior[2];

  for (let t = 0; t < T; t++) {
    const base = t * 3;
    const prevBase = (t - 1) * 3;
    if (t > 0) {
      for (let c = 0; c < 3; c++) {
        let bestVal = -Infinity;
        let bestPrev = 0;
        for (let prev = 0; prev < 3; prev++) {
          const cand = L[prevBase + prev] + logSurt[prev][c];
          if (cand > bestVal) {
            bestVal = cand;
            bestPrev = prev;
          }
        }
        ent[base + c] = bestVal;
        arg[base + c] = bestPrev;
      }
    }

    for (let c = 0; c < 3; c++) {
      const s = t - m[c];
      const nouC = s >= 0 ? ent[s * 3 + c] + C[base + c] - C[s * 3 + c] : -Infinity;
      const queda = t > 0 ? L[prevBase + c] + logQueda[c] : -Infinity;
      if (nouC > queda) {
        entrades[c].push(t);
        L[base + c] = nouC + em[base + c];
      } else {
        L[base + c] = queda + em[base + c];
      }
    }
  }

  // Backtrack
  const out = new Uint8Array(T);
  let t = T - 1;
  let c = 0;
  const lastBase = t * 3;
  if (L[lastBase + 1] > L[lastBase + c]) c = 1;
  if (L[lastBase + 2] > L[lastBase + c]) c = 2;

  while (t >= 0) {
    const list = entrades[c];
    // searchsorted right - 1
    let lo = 0;
    let hi = list.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (list[mid] <= t) lo = mid + 1;
      else hi = mid;
    }
    const k = lo - 1;
    if (k < 0 || !Number.isFinite(L[t * 3 + c])) {
      out.fill(c, 0, t + 1);
      break;
    }
    const s = list[k] - m[c];
    out.fill(c, Math.max(0, s), t + 1);
    if (s <= 0) break;
    c = arg[s * 3 + c];
    t = s - 1;
  }

  return { pred: out, wt: wtArr };
}

export function extractSpells(codes, includeDead = false) {
  const out = [];
  const T = codes.length;
  let i = 0;
  while (i < T) {
    const c = codes[i];
    let j = i + 1;
    while (j < T && codes[j] === c) j++;
    if (includeDead || c === 0 || c === 1) {
      out.push({ start: i, stop: j, code: c });
    }
    i = j;
  }
  return out;
}

export function evaluateMetrics(gtStr, predCodes) {
  const T = gtStr.length;
  const gtCodes = new Uint8Array(T);
  const cm = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  let agree = 0;
  let predDead = 0;
  for (let i = 0; i < T; i++) {
    const g = CODE_MAP[gtStr[i]] ?? 2;
    const p = predCodes[i];
    gtCodes[i] = g;
    if (g < 3 && p < 3) cm[g][p]++;
    if (g === p) agree++;
    if (p === 2) predDead++;
  }

  // Macro-F1
  let f1Sum = 0;
  for (let k = 0; k < 3; k++) {
    const tp = cm[k][k];
    const fp = cm[0][k] + cm[1][k] + cm[2][k] - tp;
    const fn = cm[k][0] + cm[k][1] + cm[k][2] - tp;
    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
    f1Sum += prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
  }
  const macroF1 = f1Sum / 3.0;

  // Spell-F1
  const gtSpells = extractSpells(gtCodes, false);
  const predSpells = extractSpells(predCodes, false);
  const overlaps = [];
  for (let i = 0; i < gtSpells.length; i++) {
    const gs = gtSpells[i];
    for (let j = 0; j < predSpells.length; j++) {
      const ps = predSpells[j];
      if (ps.code !== gs.code || ps.stop <= gs.start || gs.stop <= ps.start) continue;
      const inter = Math.min(gs.stop, ps.stop) - Math.max(gs.start, ps.start);
      const union = Math.max(gs.stop, ps.stop) - Math.min(gs.start, ps.start);
      overlaps.push([inter / union, i, j]);
    }
  }
  overlaps.sort((a, b) => (b[0] - a[0]) || (a[1] - b[1]) || (a[2] - b[2]));

  let spellF1Sum = 0;
  const spellMatchedAt50 = new Uint8Array(predSpells.length);
  for (const thr of SPELL_IOUS) {
    const usedG = new Uint8Array(gtSpells.length);
    const usedP = new Uint8Array(predSpells.length);
    let tp = 0;
    for (let idx = 0; idx < overlaps.length; idx++) {
      const [iou, gi, pi] = overlaps[idx];
      if (iou < thr) break;
      if (usedG[gi] || usedP[pi]) continue;
      usedG[gi] = 1;
      usedP[pi] = 1;
      if (thr === 0.5) spellMatchedAt50[pi] = 1;
      tp++;
    }
    const fp = predSpells.length - tp;
    const fn = gtSpells.length - tp;
    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
    spellF1Sum += prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
  }

  return {
    spellF1: spellF1Sum / SPELL_IOUS.length,
    macroF1,
    accuracy: agree / T,
    deadPct: (100 * predDead) / T,
    nGtSpells: gtSpells.length,
    nPredSpells: predSpells.length,
    gtSpells,
    predSpells,
    predAllRuns: extractSpells(predCodes, true),
    gtAllRuns: extractSpells(gtCodes, true),
    spellMatchedAt50,
  };
}
