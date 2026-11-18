import { CLASSES, CODE_MAP, decodeV9m1, evaluateMetrics } from './viterbi.js';

const COLORS = {
  H: '#8fd3ff',
  A: '#f2a99f',
  D: '#f3e2a6',
  X: '#55555c',
  0: '#8fd3ff',
  1: '#f2a99f',
  2: '#f3e2a6',
  3: '#55555c',
};

const PRESETS = {
  v9m1: { w: 1.0, factor: 0.35, lam: 0.25, s_P: 1.0, s_D: 0.25 },
  v9: { w: 1.0, factor: 0.35, lam: 0.25, s_P: 1.0, s_D: 1.0 },
  tcn: { w: 1.0, factor: 0.0, lam: 0.25, s_P: 1.0, s_D: 1.0 },
  rules: { w: 0.0, factor: 0.0, lam: 0.25, s_P: 1.0, s_D: 0.25 },
};

const SPEEDS = [0.5, 1, 2, 4];

const state = {
  manifest: [],
  scenario: null,
  cfg: { ...PRESETS.v9m1 },
  baseSpellF1: 0,
  decoded: null,
  metrics: null,
  frame: 0,
  playing: false,
  speedIdx: 1,
  zoom: 1,
  lastTick: 0,
};

const $ = (id) => document.getElementById(id);

async function init() {
  const res = await fetch('data/manifest.json');
  state.manifest = await res.json();

  const nav = $('scenarios');
  state.manifest.forEach((item, idx) => {
    const btn = document.createElement('button');
    btn.className = 'sc-btn' + (idx === 0 ? ' on' : '');
    btn.textContent = item.title;
    btn.dataset.id = item.id;
    btn.addEventListener('click', () => loadScenario(item.id));
    nav.appendChild(btn);
  });

  bindControls();
  await loadScenario(state.manifest[0].id);
  requestAnimationFrame(loop);
}

async function loadScenario(id) {
  document.querySelectorAll('.sc-btn').forEach((b) => b.classList.toggle('on', b.dataset.id === id));
  const item = state.manifest.find((m) => m.id === id);
  const res = await fetch(item.file);
  state.scenario = await res.json();
  state.frame = 0;
  state.playing = false;
  $('playBtn').textContent = '▶ Play';
  $('scenarioDesc').textContent = `${item.subtitle} · ${state.scenario.source}`;

  applyPreset('v9m1', false);
  const baseDec = decodeV9m1(state.scenario, state.scenario.defaults);
  state.baseSpellF1 = evaluateMetrics(state.scenario.signals.gt, baseDec.pred).spellF1;
  recompute();
}

function applyPreset(name, run = true) {
  const p = PRESETS[name];
  state.cfg = { ...p };
  for (const k of Object.keys(p)) {
    const sl = $('sl_' + k);
    if (sl) {
      sl.value = p[k];
      $('val_' + k).textContent = Number(p[k]).toFixed(2);
    }
  }
  document.querySelectorAll('.preset').forEach((b) => b.classList.toggle('on', b.dataset.p === name));
  if (run) recompute();
}

function recompute() {
  if (!state.scenario) return;
  const t0 = performance.now();
  state.decoded = decodeV9m1(state.scenario, state.cfg);
  state.metrics = evaluateMetrics(state.scenario.signals.gt, state.decoded.pred);
  const dt = performance.now() - t0;

  const m = state.metrics;
  const delta = m.spellF1 - state.baseSpellF1;
  const sign = delta >= 0 ? '+' : '';
  $('kpiSpell').textContent = m.spellF1.toFixed(3);
  const dEl = $('kpiSpellDelta');
  dEl.textContent = `${sign}${delta.toFixed(3)} vs v9m1`;
  dEl.className = Math.abs(delta) < 1e-4 ? '' : (delta > 0 ? 'delta-pos' : 'delta-neg');

  $('kpiMacro').textContent = m.macroF1.toFixed(3);
  $('kpiAcc').textContent = `Agreement ${(m.accuracy * 100).toFixed(1)}%`;
  $('kpiSpells').textContent = `${m.nPredSpells} / ${m.nGtSpells}`;
  $('kpiDead').textContent = `${m.deadPct.toFixed(1)}%`;
  $('kpiRuntime').textContent = `Viterbi ${dt.toFixed(1)} ms`;

  const mP = Math.max(1, Math.round(state.cfg.s_P * state.scenario.st.p5[0]));
  const mD = Math.max(1, Math.round(state.cfg.s_D * state.scenario.st.p5[2]));
  $('minLiveReadout').textContent = `H/A ≥ ${mP} frames (${(mP / 25).toFixed(2)} s)`;
  $('minDeadReadout').textContent = `D ≥ ${mD} frames (${(mD / 25).toFixed(2)} s)`;

  render();
}

function bindControls() {
  for (const k of ['w', 'factor', 'lam', 's_P', 's_D']) {
    $('sl_' + k).addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      state.cfg[k] = val;
      $('val_' + k).textContent = val.toFixed(2);
      document.querySelectorAll('.preset').forEach((b) => b.classList.remove('on'));
      recompute();
    });
  }

  document.querySelectorAll('.preset').forEach((b) => {
    b.addEventListener('click', () => applyPreset(b.dataset.p));
  });
  $('resetBtn').addEventListener('click', () => applyPreset('v9m1'));

  $('playBtn').addEventListener('click', togglePlay);
  $('backBtn').addEventListener('click', () => seek(state.frame - 125));
  $('fwdBtn').addEventListener('click', () => seek(state.frame + 125));
  $('speedBtn').addEventListener('click', () => {
    state.speedIdx = (state.speedIdx + 1) % SPEEDS.length;
    $('speedBtn').textContent = SPEEDS[state.speedIdx] + '×';
  });

  $('nextDropBtn').addEventListener('click', () => jumpNext((i) => state.scenario.signals.bVis[i] === '0'));
  $('nextErrBtn').addEventListener('click', () => {
    const gt = state.scenario.signals.gt;
    const pred = state.decoded.pred;
    jumpNext((i) => CODE_MAP[gt[i]] !== pred[i]);
  });

  document.querySelectorAll('.zbtn').forEach((b) => {
    b.addEventListener('click', () => {
      state.zoom = Number(b.dataset.z);
      document.querySelectorAll('.zbtn').forEach((z) => z.classList.toggle('on', z === b));
      drawTimeline();
    });
  });

  // Timeline scrubbing & hover
  const tc = $('timelineCanvas');
  let dragging = false;
  const xToFrame = (clientX) => {
    const rect = tc.getBoundingClientRect();
    const { winStart, winEnd } = visibleWindow();
    const frac = Math.min(1, Math.max(0, (clientX - rect.left - 84) / (rect.width - 96)));
    return Math.round(winStart + frac * (winEnd - winStart - 1));
  };
  tc.addEventListener('pointerdown', (e) => {
    dragging = true;
    tc.setPointerCapture(e.pointerId);
    seek(xToFrame(e.clientX));
  });
  tc.addEventListener('pointermove', (e) => {
    if (dragging) seek(xToFrame(e.clientX));
    showTooltip(e);
  });
  tc.addEventListener('pointerup', () => { dragging = false; });
  tc.addEventListener('pointerleave', () => { dragging = false; $('tip').hidden = true; });

  // Modal & keyboard shortcuts
  $('archBtn').addEventListener('click', () => { $('modal').hidden = false; });
  $('modalClose').addEventListener('click', () => { $('modal').hidden = true; });
  $('modal').addEventListener('click', (e) => { if (e.target === $('modal')) $('modal').hidden = true; });

  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
    else if (e.code === 'ArrowLeft') seek(state.frame - 25);
    else if (e.code === 'ArrowRight') seek(state.frame + 25);
    else if (e.key === 'm' || e.key === 'M') $('nextErrBtn').click();
    else if (e.key === 'Escape') $('modal').hidden = true;
  });

  window.addEventListener('resize', render);
}

function togglePlay() {
  state.playing = !state.playing;
  $('playBtn').textContent = state.playing ? '❚❚ Pause' : '▶ Play';
  state.lastTick = performance.now();
}

function seek(f) {
  if (!state.scenario) return;
  state.frame = Math.max(0, Math.min(state.scenario.nFrames - 1, f));
  render();
}

function jumpNext(predicate) {
  if (!state.scenario) return;
  const T = state.scenario.nFrames;
  let i = (state.frame + 10) % T;
  // Skip current contiguous block if already inside one
  while (i !== state.frame && predicate(i) && predicate((i - 1 + T) % T)) {
    i = (i + 1) % T;
  }
  for (let step = 0; step < T; step++) {
    const idx = (i + step) % T;
    if (predicate(idx)) {
      seek(idx);
      return;
    }
  }
}

function visibleWindow() {
  const T = state.scenario.nFrames;
  if (state.zoom === 1) return { winStart: 0, winEnd: T };
  const span = Math.round(T / state.zoom);
  let winStart = Math.round(state.frame - span / 2);
  winStart = Math.max(0, Math.min(T - span, winStart));
  return { winStart, winEnd: winStart + span };
}

function loop(now) {
  if (state.playing && state.scenario) {
    const dt = (now - state.lastTick) / 1000;
    state.lastTick = now;
    const advance = dt * state.scenario.fps * SPEEDS[state.speedIdx];
    state.frame += advance;
    if (state.frame >= state.scenario.nFrames) state.frame = 0;
    render();
  }
  requestAnimationFrame(loop);
}

function render() {
  if (!state.scenario || !state.decoded) return;
  const f = Math.floor(state.frame);
  const sig = state.scenario.signals;
  const predCh = CLASSES[state.decoded.pred[f]];
  const gtCh = sig.gt[f];
  const ruleCh = sig.rules[f];
  const conf = sig.conf[f] * 0.001;
  const wt = state.decoded.wt[f];
  const bVis = sig.bVis[f] === '1';

  const sec = f / state.scenario.fps;
  const mins = String(Math.floor(sec / 60)).padStart(2, '0');
  const rem = (sec % 60).toFixed(1).padStart(4, '0');
  $('clockReadout').textContent = `${mins}:${rem}`;
  $('frameReadout').textContent = `frame ${f} / ${state.scenario.nFrames}`;
  $('dropoutBadge').hidden = bVis;

  const pH = (sig.pH[f] * 0.001).toFixed(2);
  const pA = (sig.pA[f] * 0.001).toFixed(2);
  const pD = (sig.pD[f] * 0.001).toFixed(2);
  $('insTcn').textContent = `H ${pH} · A ${pA} · D ${pD}`;
  $('insRule').textContent = ruleCh;
  $('insRule').style.color = COLORS[ruleCh];
  $('insConf').textContent = conf.toFixed(2);
  $('insWt').textContent = wt.toFixed(2);
  $('insPred').textContent = predCh;
  $('insPred').style.color = COLORS[predCh];
  $('insGt').textContent = gtCh;
  $('insGt').style.color = COLORS[gtCh];

  const agree = predCh === gtCh;
  const vp = $('insVerdict');
  vp.textContent = agree ? 'MATCHES GT' : 'MISMATCH';
  vp.className = 'verdict-pill' + (agree ? '' : ' miss');

  drawPitch();
  drawTimeline();
}

function drawPitch() {
  const canvas = $('pitchCanvas');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(300, Math.floor(rect.width * dpr));
  const h = Math.max(180, Math.floor(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  const [L, W] = state.scenario.pitch;
  const pad = 18 * dpr;
  const pw = w - pad * 2;
  const ph = h - pad * 2;
  const sx = (x) => pad + (x / L) * pw;
  const sy = (y) => pad + (y / W) * ph;

  // Turf
  ctx.fillStyle = '#12161a';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(255,255,255,0.14)';
  ctx.lineWidth = 1.4 * dpr;
  ctx.strokeRect(pad, pad, pw, ph);

  // Halfway, center circle, penalty boxes
  ctx.beginPath();
  ctx.moveTo(sx(L / 2), pad);
  ctx.lineTo(sx(L / 2), pad + ph);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(sx(L / 2), sy(W / 2), (9.15 / L) * pw, 0, Math.PI * 2);
  ctx.stroke();

  const boxW = (16.5 / L) * pw;
  const boxH = (40.32 / W) * ph;
  ctx.strokeRect(pad, sy(W / 2) - boxH / 2, boxW, boxH);
  ctx.strokeRect(pad + pw - boxW, sy(W / 2) - boxH / 2, boxW, boxH);

  // Interpolated positions
  const stride = state.scenario.stride;
  const posArr = state.scenario.pos;
  const exactIdx = state.frame / stride;
  const i0 = Math.min(posArr.length - 1, Math.floor(exactIdx));
  const i1 = Math.min(posArr.length - 1, i0 + 1);
  const alpha = exactIdx - i0;

  const interp = (a, b) => (a == null ? null : b == null ? a : a + (b - a) * alpha) * 0.1;

  const f0 = posArr[i0];
  const f1 = posArr[i1];
  const teams = state.scenario.teams;

  // Players
  const players = [];
  for (let k = 0; k < teams.length; k++) {
    const px = interp(f0.p[k * 2], f1.p[k * 2]);
    const py = interp(f0.p[k * 2 + 1], f1.p[k * 2 + 1]);
    if (px == null || py == null) continue;
    const isHome = teams[k] === 0;
    players.push({ x: px, y: py, team: isHome ? 'H' : 'A' });
    ctx.beginPath();
    ctx.arc(sx(px), sy(py), 6.2 * dpr, 0, Math.PI * 2);
    ctx.fillStyle = isHome ? COLORS.H : COLORS.A;
    ctx.fill();
    ctx.lineWidth = 1.2 * dpr;
    ctx.strokeStyle = '#0c0c0e';
    ctx.stroke();
  }

  // Ball trail (last ~2 seconds)
  ctx.beginPath();
  let started = false;
  for (let back = 18; back >= 0; back--) {
    const bi = Math.max(0, i0 - back);
    const b = posArr[bi].b;
    if (b[0] == null || b[1] == null) { started = false; continue; }
    const x = sx(b[0] * 0.1);
    const y = sy(b[1] * 0.1);
    if (!started) { ctx.moveTo(x, y); started = true; }
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.28)';
  ctx.lineWidth = 1.6 * dpr;
  ctx.stroke();

  // Ball & physical contact ring
  const fInt = Math.floor(state.frame);
  const conf = state.scenario.signals.conf[fInt] * 0.001;
  const predCode = state.decoded.pred[fInt];
  const bx = interp(f0.b[0], f1.b[0]);
  const by = interp(f0.b[1], f1.b[1]);

  if (bx != null && by != null) {
    if (conf > 0.05) {
      ctx.beginPath();
      ctx.arc(sx(bx), sy(by), (11 + 12 * conf) * dpr, 0, Math.PI * 2);
      ctx.strokeStyle = COLORS[predCode];
      ctx.globalAlpha = 0.25 + 0.55 * conf;
      ctx.lineWidth = 2.2 * dpr;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.beginPath();
    ctx.arc(sx(bx), sy(by), 4.8 * dpr, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.lineWidth = 1.5 * dpr;
    ctx.strokeStyle = COLORS[predCode];
    ctx.stroke();
  } else {
    // Ball dropout marker in corner
    ctx.fillStyle = 'rgba(243, 226, 166, 0.9)';
    ctx.font = `600 ${11 * dpr}px "JetBrains Mono", monospace`;
    ctx.fillText('NO BALL TRACKING · TCN + KINEMATICS BRIDGING GAP', pad + 10 * dpr, pad + 18 * dpr);
  }
}

function drawTimeline() {
  const canvas = $('timelineCanvas');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(400, Math.floor(rect.width * dpr));
  const h = Math.max(180, Math.floor(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  const left = 84 * dpr;
  const right = w - 12 * dpr;
  const tw = right - left;
  const { winStart, winEnd } = visibleWindow();
  const span = winEnd - winStart;
  const fToX = (f) => left + ((f - winStart) / span) * tw;

  const sig = state.scenario.signals;
  const pred = state.decoded.pred;

  const rows = [
    { label: 'TCN v7d', y: 12 * dpr, rh: 44 * dpr },
    { label: 'Rules + conf', y: 66 * dpr, rh: 36 * dpr },
    { label: 'v9m1 Viterbi', y: 112 * dpr, rh: 36 * dpr },
    { label: 'Ground Truth', y: 158 * dpr, rh: 36 * dpr },
  ];

  ctx.font = `500 ${10.5 * dpr}px "JetBrains Mono", monospace`;
  ctx.textBaseline = 'middle';

  // Row labels
  rows.forEach((r) => {
    ctx.fillStyle = '#9898a0';
    ctx.fillText(r.label, 6 * dpr, r.y + r.rh / 2);
    ctx.fillStyle = '#1b1b1f';
    ctx.fillRect(left, r.y, tw, r.rh);
  });

  // Track 1: TCN stacked probabilities (sampled per pixel)
  const r0 = rows[0];
  const pxW = Math.max(1, Math.floor(tw));
  for (let px = 0; px < pxW; px++) {
    const f = Math.min(winEnd - 1, Math.floor(winStart + (px / pxW) * span));
    const hH = (sig.pH[f] * 0.001) * r0.rh;
    const hA = (sig.pA[f] * 0.001) * r0.rh;
    const hD = r0.rh - hH - hA;
    const x = left + px;
    ctx.fillStyle = 'rgba(143,211,255,0.85)';
    ctx.fillRect(x, r0.y, 1.5, hH);
    ctx.fillStyle = 'rgba(242,169,159,0.85)';
    ctx.fillRect(x, r0.y + hH, 1.5, hA);
    ctx.fillStyle = 'rgba(243,226,166,0.68)';
    ctx.fillRect(x, r0.y + hH + hA, 1.5, hD);
  }

  // Track 2: Rules step + conf_fisica line + ball dropout ticks
  const r1 = rows[1];
  for (let px = 0; px < pxW; px++) {
    const f = Math.min(winEnd - 1, Math.floor(winStart + (px / pxW) * span));
    const ch = sig.rules[f];
    ctx.fillStyle = COLORS[ch];
    ctx.globalAlpha = 0.45;
    ctx.fillRect(left + px, r1.y + r1.rh * 0.55, 1.5, r1.rh * 0.45);
    if (sig.bVis[f] === '0') {
      ctx.fillStyle = '#f3e2a6';
      ctx.globalAlpha = 0.9;
      ctx.fillRect(left + px, r1.y, 1.5, 3 * dpr);
    }
  }
  ctx.globalAlpha = 1;
  ctx.beginPath();
  for (let px = 0; px < pxW; px += 2) {
    const f = Math.min(winEnd - 1, Math.floor(winStart + (px / pxW) * span));
    const cy = r1.y + (1 - sig.conf[f] * 0.001) * (r1.rh - 6 * dpr) + 3 * dpr;
    if (px === 0) ctx.moveTo(left + px, cy);
    else ctx.lineTo(left + px, cy);
  }
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.3 * dpr;
  ctx.stroke();

  // Track 3 & 4: Spells for v9m1 and GT
  const drawRuns = (runs, row) => {
    for (const run of runs) {
      if (run.stop <= winStart || run.start >= winEnd) continue;
      const a = Math.max(winStart, run.start);
      const b = Math.min(winEnd, run.stop);
      const x0 = fToX(a);
      const x1 = fToX(b);
      ctx.fillStyle = COLORS[run.code];
      ctx.globalAlpha = run.code === 2 ? 0.52 : 0.88;
      ctx.fillRect(x0, row.y + 2 * dpr, Math.max(1.5, x1 - x0 - 1), row.rh - 4 * dpr);
    }
    ctx.globalAlpha = 1;
  };
  drawRuns(state.metrics.predAllRuns, rows[2]);
  drawRuns(state.metrics.gtAllRuns, rows[3]);

  // Mismatch strip beneath GT
  for (let px = 0; px < pxW; px++) {
    const f = Math.min(winEnd - 1, Math.floor(winStart + (px / pxW) * span));
    if (CODE_MAP[sig.gt[f]] !== pred[f]) {
      ctx.fillStyle = '#f2a99f';
      ctx.fillRect(left + px, rows[3].y + rows[3].rh + 2 * dpr, 1.5, 3 * dpr);
    }
  }

  // Playhead
  const pxCur = fToX(state.frame);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.6 * dpr;
  ctx.beginPath();
  ctx.moveTo(pxCur, 6 * dpr);
  ctx.lineTo(pxCur, h - 10 * dpr);
  ctx.stroke();
}

function showTooltip(e) {
  if (!state.scenario || !state.decoded) return;
  const tc = $('timelineCanvas');
  const rect = tc.getBoundingClientRect();
  const relX = e.clientX - rect.left;
  if (relX < 84 || relX > rect.width - 12) { $('tip').hidden = true; return; }
  const { winStart, winEnd } = visibleWindow();
  const f = Math.round(winStart + ((relX - 84) / (rect.width - 96)) * (winEnd - winStart - 1));
  const sig = state.scenario.signals;
  const pred = CLASSES[state.decoded.pred[f]];
  const gt = sig.gt[f];
  const rule = sig.rules[f];
  const sec = (f / 25).toFixed(1);
  const tip = $('tip');
  tip.innerHTML =
    `t=${sec}s (f ${f}) · <b>Pred: ${pred}</b> vs GT: ${gt}<br>` +
    `TCN H ${(sig.pH[f]/10).toFixed(0)}% A ${(sig.pA[f]/10).toFixed(0)}% D ${(sig.pD[f]/10).toFixed(0)}% · Rule: ${rule} · w_t: ${state.decoded.wt[f].toFixed(2)}`;
  tip.style.left = Math.min(rect.width - 280, Math.max(8, relX + 12)) + 'px';
  tip.style.top = '8px';
  tip.hidden = false;
}

init();
