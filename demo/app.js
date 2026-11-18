import { CODE_MAP, decodeV9m1, evaluateMetrics } from './viterbi.js';

const COLORS = { 0: '#8fd3ff', 1: '#f2a99f', 2: '#f3e2a6', 3: '#55555c' };
const NAMES = { 0: 'Home', 1: 'Away', 2: 'Play stopped', 3: 'Unknown' };
const MODES = {
  v9m1: (sc) => decodeV9m1(sc, sc.defaults),
  tcn: (sc) => {
    const { pH, pA, pD } = sc.signals;
    const out = new Uint8Array(sc.nFrames);
    for (let i = 0; i < sc.nFrames; i++) {
      out[i] = pH[i] >= pA[i] && pH[i] >= pD[i] ? 0 : (pA[i] >= pD[i] ? 1 : 2);
    }
    return { pred: out, wt: null };
  },
  rules: (sc) => {
    const r = sc.signals.rules;
    const out = new Uint8Array(sc.nFrames);
    for (let i = 0; i < sc.nFrames; i++) out[i] = CODE_MAP[r[i]] ?? 2;
    return { pred: out, wt: null };
  },
};

const SPEEDS = [1, 2, 0.5];

const state = {
  manifest: [],
  scenarios: {},
  scIdx: 0,
  sc: null,
  mode: 'v9m1',
  cache: {},
  view: 'landing',
  frame: 0,
  playing: false,
  speedIdx: 0,
  scanPhase: 0,
  lastTick: 0,
};

const $ = (id) => document.getElementById(id);

async function init() {
  const res = await fetch('data/manifest.json');
  state.manifest = await res.json();
  bindUI();
  await buildLandingCards();
  requestAnimationFrame(loop);
}

async function ensureScenario(idx) {
  const item = state.manifest[idx];
  if (!state.scenarios[item.id]) {
    const res = await fetch(item.file);
    const sc = await res.json();
    const cache = {};
    for (const m of Object.keys(MODES)) {
      const { pred, wt } = MODES[m](sc);
      cache[m] = { pred, wt, metrics: evaluateMetrics(sc.signals.gt, pred) };
    }
    state.scenarios[item.id] = { sc, cache };
  }
  return state.scenarios[item.id];
}

async function buildLandingCards() {
  const grid = $('matchGrid');
  grid.innerHTML = '';

  const loaded = await Promise.all(state.manifest.map((_, i) => ensureScenario(i)));
  state.manifest.forEach((item, idx) => {
    const { sc, cache } = loaded[idx];
    const v9 = cache.v9m1.metrics;
    const btn = document.createElement('button');
    btn.className = 'mcard';
    btn.innerHTML = `
      <div class="mcard-top">
        <b>${item.title}</b>
        <span>${Math.round(v9.spellF1 * 100)}% spell match</span>
      </div>
      <p>${item.subtitle}</p>
      <div class="mcard-bar"><canvas></canvas></div>
      <div class="mcard-foot">
        <span>${v9.nPredSpells} possessions · 4 min</span>
        <span>Open →</span>
      </div>
    `;
    btn.addEventListener('click', (e) => enterMatch(idx, e.clientX, e.clientY));
    grid.appendChild(btn);

    const cv = btn.querySelector('canvas');
    drawMiniBar(cv, sc, v9.predAllRuns, 0);
  });
}

function drawMiniBar(canvas, sc, runs, pulseFrac) {
  const w = 360;
  const h = 14;
  if (canvas.width !== w) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  const T = sc.nFrames;
  for (const r of runs) {
    const x0 = (r.start / T) * w;
    const x1 = (r.stop / T) * w;
    ctx.fillStyle = COLORS[r.code];
    ctx.globalAlpha = r.code === 2 ? 0.35 : 0.85;
    ctx.fillRect(x0, 2, Math.max(1.5, x1 - x0 - 1), h - 4);
  }
  ctx.globalAlpha = 1;
  const px = pulseFrac * w;
  const g = ctx.createLinearGradient(px - 28, 0, px + 8, 0);
  g.addColorStop(0, 'rgba(255,255,255,0)');
  g.addColorStop(1, 'rgba(255,255,255,0.65)');
  ctx.fillStyle = g;
  ctx.fillRect(px - 28, 0, 36, h);
}

async function enterMatch(idx, ox = window.innerWidth / 2, oy = window.innerHeight / 2) {
  document.body.style.setProperty('--ox', `${ox}px`);
  document.body.style.setProperty('--oy', `${oy}px`);
  await selectScenario(idx);

  state.view = 'home';
  document.body.classList.add('to-home');
  setTimeout(() => {
    $('landing').hidden = true;
    $('home').hidden = false;
    document.body.classList.remove('to-home', 'view-landing');
    renderHome();
  }, 320);
}

function backToLanding() {
  state.view = 'landing';
  $('home').hidden = true;
  $('landing').hidden = false;
  document.body.classList.add('view-landing');
}

async function selectScenario(idx) {
  state.scIdx = idx % state.manifest.length;
  const item = state.manifest[state.scIdx];
  const { sc, cache } = await ensureScenario(state.scIdx);
  state.sc = sc;
  state.cache = cache;
  state.frame = findFirstHighlight();
  $('matchName').textContent = item.title;
  $('diveTitle').textContent = item.title;
}

function active() {
  return state.cache[state.mode];
}

function setMode(m) {
  state.mode = m;
  document.querySelectorAll('#modeSeg button, #diveSeg button').forEach((b) => {
    b.classList.toggle('on', b.dataset.m === m);
  });
  if (state.view === 'home') renderHome();
  if (state.view === 'dive') renderDive();
}

function renderHome() {
  if (!state.sc) return;
  const { metrics } = active();
  const label = state.mode === 'v9m1' ? 'Combined model' : (state.mode === 'tcn' ? 'Context only' : 'Physics only');
  const degNote = state.sc.degraded ? 'with 40% missing ball tracking' : 'on clean tracking';
  $('sum').innerHTML =
    `<b>${label}</b> matches <b>${Math.round(metrics.spellF1 * 100)}%</b> of real possessions ` +
    `(<b>${metrics.nPredSpells}</b> detected vs <b>${metrics.nGtSpells}</b> actual) ${degNote}. Click any bar to watch it happen.`;
  drawStream();
}

function drawStream() {
  const svg = $('stream');
  const W = Math.max(760, svg.clientWidth || 1000);
  const H = 230;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = '';

  const T = state.sc.nFrames;
  const xOf = (f) => (f / T) * W;
  const ns = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, parent = svg) => {
    const n = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    parent.appendChild(n);
    return n;
  };

  const addTag = (y, txt) => {
    const t = el('text', { x: 0, y, fill: '#55555c', 'font-size': '10.5', 'font-family': 'JetBrains Mono, monospace' });
    t.textContent = txt;
  };
  addTag(28, 'REAL MATCH');
  addTag(82, state.mode === 'v9m1' ? 'COMBINED MODEL' : (state.mode === 'tcn' ? 'CONTEXT ONLY' : 'PHYSICS ONLY'));
  addTag(172, 'TOUCH CONFIDENCE');

  for (const r of active().metrics.gtAllRuns) {
    el('rect', {
      x: xOf(r.start).toFixed(1),
      y: 38,
      width: Math.max(1.5, xOf(r.stop) - xOf(r.start) - 1).toFixed(1),
      height: 8,
      rx: 4,
      fill: COLORS[r.code],
      opacity: r.code === 2 ? '0.28' : '0.42',
    });
  }

  for (const r of active().metrics.predAllRuns) {
    const x0 = xOf(r.start);
    const x1 = xOf(r.stop);
    const rect = el('rect', {
      class: 'spell',
      x: x0.toFixed(1),
      y: 94,
      width: Math.max(2.5, x1 - x0 - 1.5).toFixed(1),
      height: 36,
      rx: 7,
      fill: COLORS[r.code],
      opacity: r.code === 2 ? '0.45' : '0.92',
    });
    rect.addEventListener('mouseenter', (e) => showTip(e, r));
    rect.addEventListener('mousemove', moveTip);
    rect.addEventListener('mouseleave', () => { $('tip').hidden = true; });
    rect.addEventListener('click', (e) => {
      $('tip').hidden = true;
      openDive(r.start, e.clientX, e.clientY);
    });
  }

  const bVis = state.sc.signals.bVis;
  let i = 0;
  while (i < T) {
    if (bVis[i] === '0') {
      let j = i + 1;
      while (j < T && bVis[j] === '0') j++;
      if (j - i > 6) {
        el('line', {
          x1: xOf(i).toFixed(1), y1: 142, x2: xOf(j).toFixed(1), y2: 142,
          stroke: '#f3e2a6', 'stroke-width': '2', 'stroke-dasharray': '3 3', opacity: '0.65',
        });
      }
      i = j;
    } else i++;
  }

  const conf = state.sc.signals.conf;
  const pts = [];
  for (let f = 0; f < T; f += 12) {
    pts.push(`${xOf(f).toFixed(1)},${(214 - (conf[f] * 0.001) * 28).toFixed(1)}`);
  }
  el('polyline', { fill: 'none', stroke: '#b7a4ff', 'stroke-width': '1.4', opacity: '0.58', points: pts.join(' ') });
}

function showTip(e, run) {
  const dur = ((run.stop - run.start) / state.sc.fps).toFixed(1);
  let dropFrames = 0;
  for (let f = run.start; f < run.stop; f++) if (state.sc.signals.bVis[f] === '0') dropFrames++;
  const dropTxt = dropFrames > 10 ? ` · ${Math.round((100 * dropFrames) / (run.stop - run.start))}% blind` : '';
  const tip = $('tip');
  tip.innerHTML = `<b style="color:${COLORS[run.code]}">${NAMES[run.code]}</b> · ${dur} s<small>${fmtTime(run.start)}${dropTxt} · click to watch</small>`;
  tip.hidden = false;
  moveTip(e);
}

function moveTip(e) {
  const tip = $('tip');
  tip.style.left = Math.min(window.innerWidth - 220, e.clientX + 14) + 'px';
  tip.style.top = Math.max(12, e.clientY - 52) + 'px';
}

function openDive(startFrame = state.frame, ox = window.innerWidth / 2, oy = window.innerHeight / 2) {
  document.body.style.setProperty('--ox', `${ox}px`);
  document.body.style.setProperty('--oy', `${oy}px`);
  state.frame = startFrame;
  state.view = 'dive';
  state.playing = true;
  state.lastTick = performance.now();
  $('playBtn').textContent = '❚❚';

  document.body.classList.add('dive');
  const full = $('full');
  full.hidden = false;
  requestAnimationFrame(() => full.classList.add('on'));
  renderDive();
}

function closeDive() {
  state.playing = false;
  state.view = 'home';
  const full = $('full');
  full.classList.remove('on');
  document.body.classList.remove('dive');
  setTimeout(() => { if (state.view !== 'dive') full.hidden = true; }, 380);
}

function findFirstHighlight() {
  const bVis = state.sc.signals.bVis;
  for (let f = 150; f < state.sc.nFrames - 200; f++) {
    if (bVis[f] === '0' && bVis[f + 25] === '0') return Math.max(0, f - 40);
  }
  return 0;
}

function nextHighlight() {
  const T = state.sc.nFrames;
  const bVis = state.sc.signals.bVis;
  const conf = state.sc.signals.conf;
  let f = (Math.floor(state.frame) + 75) % T;
  for (let s = 0; s < T; s++) {
    const idx = (f + s) % T;
    if ((bVis[idx] === '0' && bVis[(idx + 15) % T] === '0') || conf[idx] > 780) {
      state.frame = Math.max(0, idx - 30);
      renderDive();
      return;
    }
  }
}

function renderDive() {
  if (!state.sc) return;
  const f = Math.floor(state.frame);
  const sig = state.sc.signals;
  const pred = active().pred[f];
  const gt = CODE_MAP[sig.gt[f]] ?? 2;
  const agree = pred === gt;

  $('timeTxt').textContent = fmtTime(f);
  $('diveSub').textContent = `${fmtTime(f)} · ${NAMES[pred]}`;
  $('dropPill').hidden = sig.bVis[f] === '1';

  const vEl = $('fverdict');
  vEl.textContent = agree ? 'Matches real play' : `Actual: ${NAMES[gt]}`;
  vEl.className = 'fverdict' + (agree ? '' : ' miss');

  const probs = [sig.pH[f], sig.pA[f], sig.pD[f]];
  const tcnArg = probs[0] >= probs[1] && probs[0] >= probs[2] ? 0 : (probs[1] >= probs[2] ? 1 : 2);
  const tcnPct = Math.round(probs[tcnArg] / 10);
  const ruleCode = CODE_MAP[sig.rules[f]] ?? 2;
  const conf = sig.conf[f] * 0.001;
  const wt = state.mode === 'tcn' ? 1.0 : (state.mode === 'rules' ? 0.0 : (state.cache.v9m1.wt ? state.cache.v9m1.wt[f] : 1.0));
  $('wtFill').style.width = `${Math.round(wt * 100)}%`;

  let reason;
  if (state.mode === 'tcn') {
    reason = `Context AI alone · predicts <b style="color:${COLORS[tcnArg]}">${NAMES[tcnArg]} (${tcnPct}%)</b> without physics or smoothing`;
  } else if (state.mode === 'rules') {
    reason = sig.bVis[f] === '0'
      ? `Ball missing · <b>Physics alone</b> keeps previous team (${NAMES[ruleCode]})`
      : `Touch physics alone · sees <b style="color:${COLORS[ruleCode]}">${NAMES[ruleCode]}</b> controlling the ball`;
  } else if (sig.bVis[f] === '0') {
    reason = `Ball missing in tracking · <b>20 s context</b> bridges the gap → <b style="color:${COLORS[pred]}">${NAMES[pred]} (${tcnPct}%)</b>`;
  } else if (conf > 0.35 && ruleCode !== tcnArg) {
    reason = `Clean touch detected · <b>physics takes priority</b> over context → <b style="color:${COLORS[pred]}">${NAMES[pred]}</b>`;
  } else if (tcnArg === ruleCode && pred === tcnArg) {
    reason = `Context (${tcnPct}%) &amp; touch physics agree → <b style="color:${COLORS[pred]}">${NAMES[pred]}</b>`;
  } else if (pred !== tcnArg && pred !== ruleCode) {
    reason = `Smoothing ignores a brief flicker · keeps <b style="color:${COLORS[pred]}">${NAMES[pred]}</b>`;
  } else {
    reason = `Context leans ${NAMES[tcnArg]} (${tcnPct}%), physics leans ${NAMES[ruleCode]} → <b style="color:${COLORS[pred]}">${NAMES[pred]}</b>`;
  }
  $('decisionLine').innerHTML = reason;

  drawPitch();
  drawSignals();
  drawScrub();
}

function drawPitch() {
  const canvas = $('pitch');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(300, Math.floor(rect.width * dpr));
  const h = Math.max(200, Math.floor(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  const [L, W] = state.sc.pitch;
  const pad = 20 * dpr;
  const pw = w - pad * 2;
  const ph = h - pad * 2;
  const sx = (x) => pad + (x / L) * pw;
  const sy = (y) => pad + (y / W) * ph;

  ctx.fillStyle = '#131417';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1.4 * dpr;
  ctx.strokeRect(pad, pad, pw, ph);

  ctx.beginPath();
  ctx.moveTo(sx(L / 2), pad);
  ctx.lineTo(sx(L / 2), pad + ph);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(sx(L / 2), sy(W / 2), (9.15 / L) * pw, 0, Math.PI * 2);
  ctx.stroke();

  const bw = (16.5 / L) * pw;
  const bh = (40.32 / W) * ph;
  ctx.strokeRect(pad, sy(W / 2) - bh / 2, bw, bh);
  ctx.strokeRect(pad + pw - bw, sy(W / 2) - bh / 2, bw, bh);

  const stride = state.sc.stride;
  const posArr = state.sc.pos;
  const exact = state.frame / stride;
  const i0 = Math.min(posArr.length - 1, Math.floor(exact));
  const i1 = Math.min(posArr.length - 1, i0 + 1);
  const alpha = exact - i0;
  const interp = (a, b) => (a == null ? null : (b == null ? a : a + (b - a) * alpha) * 0.1);

  const f0 = posArr[i0];
  const f1 = posArr[i1];
  const teams = state.sc.teams;
  for (let k = 0; k < teams.length; k++) {
    const px = interp(f0.p[k * 2], f1.p[k * 2]);
    const py = interp(f0.p[k * 2 + 1], f1.p[k * 2 + 1]);
    if (px == null || py == null) continue;
    ctx.beginPath();
    ctx.arc(sx(px), sy(py), 6.2 * dpr, 0, Math.PI * 2);
    ctx.fillStyle = teams[k] === 0 ? COLORS[0] : COLORS[1];
    ctx.fill();
  }

  const fInt = Math.floor(state.frame);
  const conf = state.sc.signals.conf[fInt] * 0.001;
  const pred = active().pred[fInt];
  const bx = interp(f0.b[0], f1.b[0]);
  const by = interp(f0.b[1], f1.b[1]);

  ctx.beginPath();
  let moved = false;
  for (let back = 14; back >= 0; back--) {
    const b = posArr[Math.max(0, i0 - back)].b;
    if (b[0] == null || b[1] == null) { moved = false; continue; }
    const x = sx(b[0] * 0.1);
    const y = sy(b[1] * 0.1);
    if (!moved) { ctx.moveTo(x, y); moved = true; } else ctx.lineTo(x, y);
  }
  if (bx != null && by != null && moved) ctx.lineTo(sx(bx), sy(by));
  ctx.strokeStyle = 'rgba(255,255,255,0.28)';
  ctx.lineWidth = 1.6 * dpr;
  ctx.stroke();

  if (bx != null && by != null) {
    if (conf > 0.08) {
      ctx.beginPath();
      ctx.arc(sx(bx), sy(by), (12 + 11 * conf) * dpr, 0, Math.PI * 2);
      ctx.strokeStyle = COLORS[pred];
      ctx.globalAlpha = 0.28 + 0.52 * conf;
      ctx.lineWidth = 2.2 * dpr;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.beginPath();
    ctx.arc(sx(bx), sy(by), 5 * dpr, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
  }
}

function drawSignals() {
  const canvas = $('signals');
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(300, Math.floor(rect.width * dpr));
  const h = Math.max(70, Math.floor(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  const left = 64 * dpr;
  const tw = w - left - 4 * dpr;
  const T = state.sc.nFrames;
  const halfWin = 500;
  const fStart = state.frame - halfWin;
  const fSpan = halfWin * 2;
  const fToX = (f) => left + ((f - fStart) / fSpan) * tw;

  const sig = state.sc.signals;
  const pred = active().pred;
  const rows = [
    { lbl: 'Context', y: 4 * dpr, rh: 26 * dpr },
    { lbl: 'Physics', y: 36 * dpr, rh: 22 * dpr },
    { lbl: 'Final', y: 64 * dpr, rh: 20 * dpr },
  ];

  ctx.font = `500 ${10 * dpr}px "JetBrains Mono", monospace`;
  ctx.textBaseline = 'middle';
  rows.forEach((r) => {
    ctx.fillStyle = '#63636c';
    ctx.fillText(r.lbl, 0, r.y + r.rh / 2);
    ctx.fillStyle = 'rgba(255,255,255,0.03)';
    ctx.fillRect(left, r.y, tw, r.rh);
  });

  const f0 = Math.max(0, Math.floor(fStart));
  const f1 = Math.min(T - 1, Math.ceil(fStart + fSpan));
  const stepF = Math.max(1, Math.floor(fSpan / tw));
  for (let f = f0; f <= f1; f += stepF) {
    const x0 = Math.max(left, fToX(f));
    const x1 = Math.min(left + tw, fToX(f + stepF));
    const cw = Math.max(1, x1 - x0);

    const r0 = rows[0];
    const hH = (sig.pH[f] * 0.001) * r0.rh;
    const hA = (sig.pA[f] * 0.001) * r0.rh;
    ctx.fillStyle = 'rgba(143,211,255,0.82)';
    ctx.fillRect(x0, r0.y, cw, hH);
    ctx.fillStyle = 'rgba(242,169,159,0.82)';
    ctx.fillRect(x0, r0.y + hH, cw, hA);
    ctx.fillStyle = 'rgba(243,226,166,0.55)';
    ctx.fillRect(x0, r0.y + hH + hA, cw, r0.rh - hH - hA);

    const r1 = rows[1];
    const rc = CODE_MAP[sig.rules[f]] ?? 2;
    ctx.fillStyle = COLORS[rc];
    ctx.globalAlpha = 0.42;
    ctx.fillRect(x0, r1.y + r1.rh * 0.58, cw, r1.rh * 0.42);
    if (sig.bVis[f] === '0') {
      ctx.fillStyle = COLORS[2];
      ctx.globalAlpha = 0.9;
      ctx.fillRect(x0, r1.y, cw, 2.5 * dpr);
    }

    const r2 = rows[2];
    const pCode = pred[f];
    ctx.fillStyle = COLORS[pCode];
    ctx.globalAlpha = pCode === 2 ? 0.5 : 0.9;
    ctx.fillRect(x0, r2.y + 2 * dpr, cw, r2.rh - 5 * dpr);
    if ((CODE_MAP[sig.gt[f]] ?? 2) !== pCode) {
      ctx.fillStyle = '#f2a99f';
      ctx.globalAlpha = 0.9;
      ctx.fillRect(x0, r2.y + r2.rh - 2 * dpr, cw, 2 * dpr);
    }
  }
  ctx.globalAlpha = 1;

  ctx.beginPath();
  let pen = false;
  for (let f = f0; f <= f1; f += 2) {
    const x = fToX(f);
    if (x < left || x > left + tw) { pen = false; continue; }
    const cy = rows[1].y + (1 - sig.conf[f] * 0.001) * (rows[1].rh - 4 * dpr) + 2 * dpr;
    if (!pen) { ctx.moveTo(x, cy); pen = true; } else ctx.lineTo(x, cy);
  }
  ctx.strokeStyle = '#b7a4ff';
  ctx.lineWidth = 1.4 * dpr;
  ctx.stroke();

  const cx = left + tw / 2;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5 * dpr;
  ctx.beginPath();
  ctx.moveTo(cx, 2 * dpr);
  ctx.lineTo(cx, h - 2 * dpr);
  ctx.stroke();
}

function drawScrub() {
  const canvas = $('scrub');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(120, Math.floor(rect.width * dpr));
  const h = Math.max(16, Math.floor(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  const T = state.sc.nFrames;
  for (const r of active().metrics.predAllRuns) {
    const x0 = (r.start / T) * w;
    const x1 = (r.stop / T) * w;
    ctx.fillStyle = COLORS[r.code];
    ctx.globalAlpha = r.code === 2 ? 0.35 : 0.78;
    ctx.fillRect(x0, 5 * dpr, Math.max(1.5, x1 - x0 - 1), h - 10 * dpr);
  }
  ctx.globalAlpha = 1;
  const px = (state.frame / T) * w;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(px - 1.5 * dpr, 1 * dpr, 3 * dpr, h - 2 * dpr);
}

function fmtTime(f) {
  const sec = Math.floor(f / (state.sc?.fps || 25));
  return `${String(Math.floor(sec / 60)).padStart(2, '0')}:${String(sec % 60).padStart(2, '0')}`;
}

function bindUI() {
  document.querySelectorAll('.pnode').forEach((n) => {
    n.addEventListener('mouseenter', () => { $('pipeNote').textContent = n.dataset.note; });
  });
  $('pipeSvg').addEventListener('mouseleave', () => {
    $('pipeNote').textContent = 'Hover any stage to see what it does · choose a match clip below';
  });

  $('toLanding').addEventListener('click', backToLanding);
  $('matchBtn').addEventListener('click', async () => {
    await selectScenario(state.scIdx + 1);
    renderHome();
  });
  $('openDive').addEventListener('click', (e) => openDive(state.frame, e.clientX, e.clientY));
  $('back').addEventListener('click', closeDive);

  document.querySelectorAll('#modeSeg button, #diveSeg button').forEach((b) => {
    b.addEventListener('click', () => setMode(b.dataset.m));
  });

  $('playBtn').addEventListener('click', () => {
    state.playing = !state.playing;
    state.lastTick = performance.now();
    $('playBtn').textContent = state.playing ? '❚❚' : '▶';
  });
  $('speedBtn').addEventListener('click', () => {
    state.speedIdx = (state.speedIdx + 1) % SPEEDS.length;
    $('speedBtn').textContent = SPEEDS[state.speedIdx] + '×';
  });
  $('nextBtn').addEventListener('click', nextHighlight);

  const sw = $('scrubWrap');
  let drag = false;
  const seekEv = (e) => {
    const rect = sw.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    state.frame = Math.round(frac * (state.sc.nFrames - 1));
    renderDive();
  };
  sw.addEventListener('pointerdown', (e) => { drag = true; sw.setPointerCapture(e.pointerId); seekEv(e); });
  sw.addEventListener('pointermove', (e) => { if (drag) seekEv(e); });
  sw.addEventListener('pointerup', () => { drag = false; });

  const sigEl = $('signals');
  let sigDrag = false;
  let sigStartX = 0;
  let sigStartFrame = 0;
  sigEl.addEventListener('pointerdown', (e) => {
    sigDrag = true;
    sigStartX = e.clientX;
    sigStartFrame = state.frame;
    sigEl.setPointerCapture(e.pointerId);
  });
  sigEl.addEventListener('pointermove', (e) => {
    if (!sigDrag || !state.sc) return;
    const dx = e.clientX - sigStartX;
    const framesPerPx = 1000 / Math.max(200, sigEl.getBoundingClientRect().width - 64);
    state.frame = Math.max(0, Math.min(state.sc.nFrames - 1, Math.round(sigStartFrame - dx * framesPerPx)));
    renderDive();
  });
  sigEl.addEventListener('pointerup', () => { sigDrag = false; });

  const openModal = () => { $('modal').hidden = false; };
  $('infoBtn').addEventListener('click', openModal);
  $('infoBtn2').addEventListener('click', openModal);
  $('mx').addEventListener('click', () => { $('modal').hidden = true; });
  $('modal').addEventListener('click', (e) => { if (e.target === $('modal')) $('modal').hidden = true; });

  window.addEventListener('keydown', (e) => {
    if (e.code === 'Escape') {
      if (!$('modal').hidden) $('modal').hidden = true;
      else if (state.view === 'dive') closeDive();
      else if (state.view === 'home') backToLanding();
    } else if (state.view === 'dive' && e.code === 'Space') {
      e.preventDefault();
      $('playBtn').click();
    }
  });
  window.addEventListener('resize', () => {
    if (state.view === 'home') renderHome();
    if (state.view === 'dive') renderDive();
  });
}

function loop(now) {
  if (state.view === 'landing') {
    state.scanPhase = (now * 0.00018) % 1;
    const bars = document.querySelectorAll('.mcard-bar canvas');
    state.manifest.forEach((item, idx) => {
      const entry = state.scenarios[item.id];
      const cv = bars[idx];
      if (entry && cv) drawMiniBar(cv, entry.sc, entry.cache.v9m1.metrics.predAllRuns, (state.scanPhase + idx * 0.33) % 1);
    });
  } else if (state.playing && state.view === 'dive' && state.sc) {
    const dt = Math.min(0.1, (now - state.lastTick) / 1000);
    state.lastTick = now;
    state.frame += dt * state.sc.fps * SPEEDS[state.speedIdx];
    if (state.frame >= state.sc.nFrames) state.frame = 0;
    renderDive();
  }
  requestAnimationFrame(loop);
}

init();
