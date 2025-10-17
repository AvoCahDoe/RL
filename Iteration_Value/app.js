// MDP Average Gain Solver (Policy Iteration)
// CSV format: state,action,next_state,prob,reward

function parseCSV(text) {
  const rows = [];
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  for (const line of lines) {
    const cells = line.split(',').map(c => c.trim());
    if (cells.length < 5) continue;
    if (isNaN(Number(cells[3])) || isNaN(Number(cells[4]))) continue;
    if (cells[0].toLowerCase() === 'state' || cells[0].toLowerCase() === 'etat') continue; // header (EN/FR)
    rows.push({ s: cells[0], a: cells[1], sp: cells[2], p: Number(cells[3]), r: Number(cells[4]) });
  }
  return rows;
}

function buildModel(rows) {
  // states and actions
  const stateSet = new Set();
  const actionsByState = new Map();
  const P = new Map(); // key: `${s}|${a}|${sp}` -> prob
  const R = new Map(); // key: `${s}|${a}` -> expected reward
  const PSum = new Map(); // key: `${s}|${a}` -> sum prob

  for (const { s, a, sp, p, r } of rows) {
    stateSet.add(s); stateSet.add(sp);
    if (!actionsByState.has(s)) actionsByState.set(s, new Set());
    actionsByState.get(s).add(a);
    const key = `${s}|${a}|${sp}`;
    P.set(key, (P.get(key) || 0) + p);
    const raKey = `${s}|${a}`;
    R.set(raKey, (R.get(raKey) || 0) + p * r);
    PSum.set(raKey, (PSum.get(raKey) || 0) + p);
  }
  const states = Array.from(stateSet).sort();
  // Validate prob sums
  for (const [raKey, total] of PSum.entries()) {
    if (Math.abs(total - 1) > 1e-8) {
      throw new Error(`Probabilities for (${raKey.replace('|', ',')}) do not sum to 1: ${total}`);
    }
  }
  return { states, actionsByState, P, R };
}

function makeMatrix(n, fill = 0) {
  return Array.from({ length: n }, () => Array(n).fill(fill));
}

function solveLinear(A, b) {
  // Gaussian elimination with partial pivoting
  const n = A.length;
  // Augment
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    // pivot
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < 1e-12) continue; // singular, keep moving
    if (pivot !== col) { const tmp = M[col]; M[col] = M[pivot]; M[pivot] = tmp; }
    // normalize
    const pv = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= pv;
    // eliminate
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      if (Math.abs(factor) < 1e-12) continue;
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }
  // extract
  return M.map(row => row[n]);
}

function stationaryDistribution(Ppi) {
  const n = Ppi.length;
  // Solve (I - P^T) d = 0 with sum d = 1
  const A = makeMatrix(n);
  const b = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      A[i][j] = (i === j ? 1 : 0) - Ppi[j][i]; // (I - P^T)
    }
  }
  // Replace last equation with sum d = 1
  for (let j = 0; j < n; j++) A[n - 1][j] = 1;
  b[n - 1] = 1;
  return solveLinear(A, b);
}

function evaluatePolicy(model, policy) {
  const { states, actionsByState, P, R } = model;
  const n = states.length;
  const idx = new Map(states.map((s, i) => [s, i]));
  const Ppi = makeMatrix(n, 0);
  const rpi = Array(n).fill(0);

  for (const s of states) {
    const a = policy.get(s);
    const i = idx.get(s);
    const raKey = `${s}|${a}`;
    rpi[i] = R.get(raKey) || 0;
    // transitions from (s,a)
    for (const sp of states) {
      const key = `${s}|${a}|${sp}`;
      const p = P.get(key) || 0;
      const j = idx.get(sp);
      Ppi[i][j] += p;
    }
  }
  const d = stationaryDistribution(Ppi);
  const g = rpi.reduce((acc, rv, i) => acc + d[i] * rv, 0);
  // Solve (I - Ppi) h = rpi - g 1 with h(s0)=0
  const A = makeMatrix(n, 0);
  const b = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      A[i][j] = (i === j ? 1 : 0) - Ppi[i][j];
    }
    b[i] = rpi[i] - g;
  }
  // Fix h(s0)=0 by replacing row 0
  for (let j = 0; j < n; j++) A[0][j] = j === 0 ? 1 : 0;
  b[0] = 0;
  const h = solveLinear(A, b);
  return { Ppi, rpi, d, g, h };
}

function improvePolicy(model, eval) {
  const { states, actionsByState, P, R } = model;
  const n = states.length;
  const idx = new Map(states.map((s, i) => [s, i]));
  const { g, h } = eval;
  const newPolicy = new Map();
  let changed = false;

  for (const s of states) {
    const actions = Array.from(actionsByState.get(s));
    let bestA = actions[0];
    let bestQ = -Infinity;
    for (const a of actions) {
      const raKey = `${s}|${a}`;
      const expR = R.get(raKey) || 0;
      let sumNext = 0;
      for (const sp of states) {
        const key = `${s}|${a}|${sp}`;
        const p = P.get(key) || 0;
        const j = idx.get(sp);
        sumNext += p * h[j];
      }
      const Q = expR - g + sumNext;
      if (Q > bestQ + 1e-12) { bestQ = Q; bestA = a; }
    }
    newPolicy.set(s, bestA);
  }
  return newPolicy;
}

function policiesEqual(p1, p2) {
  if (p1.size !== p2.size) return false;
  for (const [s, a] of p1.entries()) {
    if (p2.get(s) !== a) return false;
  }
  return true;
}

function policyIteration(model) {
  const { states, actionsByState } = model;
  let policy = new Map();
  // initial policy: highest expected reward action
  for (const s of states) {
    const actions = Array.from(actionsByState.get(s));
    let bestA = actions[0];
    let bestR = -Infinity;
    for (const a of actions) {
      const expR = model.R.get(`${s}|${a}`) || 0;
      if (expR > bestR + 1e-12) { bestR = expR; bestA = a; }
    }
    policy.set(s, bestA);
  }
  let eval = evaluatePolicy(model, policy);
  for (let iter = 0; iter < 50; iter++) {
    const newPolicy = improvePolicy(model, eval);
    if (policiesEqual(newPolicy, policy)) break;
    policy = newPolicy;
    eval = evaluatePolicy(model, policy);
  }
  return { policy, ...eval };
}

function valueIteration(model, gamma = 0.95, theta = 1e-8, maxIter = 10000) {
  const { states, actionsByState, P, R } = model;
  const idx = new Map(states.map((s, i) => [s, i]));
  const V = Array(states.length).fill(0);
  let iters = 0;
  while (iters < maxIter) {
    let delta = 0;
    for (const s of states) {
      const actions = Array.from(actionsByState.get(s) || []);
      if (!actions.length) continue;
      let bestQ = -Infinity;
      for (const a of actions) {
        const expR = R.get(`${s}|${a}`) || 0; // E[r | s,a]
        let sumNext = 0;
        for (const sp of states) {
          const key = `${s}|${a}|${sp}`;
          const p = P.get(key) || 0;
          const j = idx.get(sp);
          sumNext += p * V[j];
        }
        const Q = expR + gamma * sumNext;
        if (Q > bestQ) bestQ = Q;
      }
      const i = idx.get(s);
      delta = Math.max(delta, Math.abs(bestQ - V[i]));
      V[i] = bestQ;
    }
    iters++;
    if (delta < theta) break;
  }
  // greedy policy from V
  const policy = new Map();
  for (const s of states) {
    const actions = Array.from(actionsByState.get(s) || []);
    if (!actions.length) continue;
    let bestA = actions[0];
    let bestQ = -Infinity;
    for (const a of actions) {
      const expR = R.get(`${s}|${a}`) || 0;
      let sumNext = 0;
      for (const sp of states) {
        const key = `${s}|${a}|${sp}`;
        const p = P.get(key) || 0;
        const j = idx.get(sp);
        sumNext += p * V[j];
      }
      const Q = expR + gamma * sumNext;
      if (Q > bestQ) { bestQ = Q; bestA = a; }
    }
    policy.set(s, bestA);
  }
  return { V, policy, iterations: iters };
}

function formatVector(states, vec, decimals = 6) {
  let html = '<table class="table"><thead><tr><th>State</th><th>Value</th></tr></thead><tbody>';
  states.forEach((s, i) => { html += `<tr><td>${s}</td><td>${Number(vec[i].toFixed(decimals))}</td></tr>`; });
  html += '</tbody></table>';
  return html;
}

function formatPolicy(states, policy) {
  let html = '<table class="table"><thead><tr><th>State</th><th>Optimal Action</th></tr></thead><tbody>';
  states.forEach(s => { html += `<tr><td>${s}</td><td>${policy.get(s)}</td></tr>`; });
  html += '</tbody></table>';
  return html;
}

function loadSample() {
  const sample = [
    'state,action,next_state,prob,reward',
    's0,go,s1,0.8,0',
    's0,go,s0,0.2,-0.2',
    's0,stay,s0,1.0,-0.05',
    's1,go,s2,0.9,1.0',
    's1,go,s0,0.1,0.0',
    's1,stay,s1,1.0,-0.05',
    's2,stay,s2,1.0,0.0',
    's2,go,s2,1.0,0.0',
  ].join('\n');
  document.getElementById('csvInput').value = sample;
}

// Visualization helpers
let currentModel = null;
let currentResult = null;
let vizListenersAttached = false;

function getUniqueActions(model) {
  const { states, actionsByState } = model;
  const set = new Set();
  for (const s of states) {
    (actionsByState.get(s) || new Set()).forEach(a => set.add(a));
  }
  return Array.from(set).sort();
}

function populateActionFilter(actions) {
  const sel = document.getElementById('actionFilter');
  if (!sel) return;
  sel.innerHTML = '<option value="__ALL__">All</option>' + actions.map(a => `<option value="${a}">${a}</option>`).join('');
}

function actionColorMap(actions) {
  const palette = ['#ef4444','#10b981','#3b82f6','#f59e0b','#8b5cf6','#14b8a6','#dc2626','#16a34a','#1d4ed8','#f97316','#7c3aed','#0ea5e9'];
  const map = new Map();
  actions.forEach((a, i) => map.set(a, palette[i % palette.length]));
  return map;
}

function renderMDPGraph() {
  const container = document.getElementById('vizContainer');
  if (!container || !currentModel) return;
  container.innerHTML = '';
  const { states, actionsByState, P, R } = currentModel;
  const { policy, d } = currentResult || { policy: new Map(), d: [] };
  const actions = getUniqueActions(currentModel);
  const colors = actionColorMap(actions);

  const filterSel = document.getElementById('actionFilter');
  const chosen = filterSel ? filterSel.value : '__ALL__';
  const policyOnly = document.getElementById('policyOnly')?.checked || false;

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'mdp-graph');
  const width = container.clientWidth || 900;
  const height = 400;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

  // Arrowhead definition
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
  marker.setAttribute('id', 'arrow');
  marker.setAttribute('markerWidth', '10');
  marker.setAttribute('markerHeight', '10');
  marker.setAttribute('refX', '8');
  marker.setAttribute('refY', '3');
  marker.setAttribute('orient', 'auto');
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', 'M0,0 L8,3 L0,6 Z');
  path.setAttribute('fill', '#6b7280');
  marker.appendChild(path);
  defs.appendChild(marker);
  svg.appendChild(defs);

  // Layout: circle
  const n = states.length;
  const cx = width / 2, cy = height / 2, radius = Math.min(width, height) * 0.35;
  const idx = new Map(states.map((s, i) => [s, i]));
  const pos = states.map((s, i) => {
    const ang = (2 * Math.PI * i) / n;
    return { x: cx + radius * Math.cos(ang), y: cy + radius * Math.sin(ang) };
  });

  // Draw edges
  for (const s of states) {
    const sIdx = idx.get(s);
    const actions = Array.from(actionsByState.get(s) || []);
    const chosenPolicyA = policy.get(s);
    for (const a of actions) {
      if (policyOnly && a !== chosenPolicyA) continue;
      if (chosen !== '__ALL__' && a !== chosen) continue;
      for (const sp of states) {
        const key = `${s}|${a}|${sp}`;
        const p = P.get(key) || 0;
        if (p <= 0) continue;
        const color = colors.get(a) || '#6b7280';
        const from = pos[sIdx];
        const to = pos[idx.get(sp)];
        const rNode = 20 + (d[sIdx] ? Math.max(0, d[sIdx]) * 20 : 0);
        if (s === sp) {
          // self-loop: small arc near node
          const loopR = rNode + 12;
          const loop = document.createElementNS('http://www.w3.org/2000/svg', 'path');
          const startX = from.x, startY = from.y - loopR;
          const endX = from.x + 1, endY = from.y - loopR + 1; // close to start
          const dpath = `M ${startX} ${startY} a ${loopR} ${loopR} 0 1 1 ${endX - startX} ${endY - startY}`;
          loop.setAttribute('d', dpath);
          loop.setAttribute('class', 'edge');
          loop.setAttribute('stroke', color);
          svg.appendChild(loop);
          const lbl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          lbl.setAttribute('class', 'edge-label');
          lbl.setAttribute('x', startX - 10);
          lbl.setAttribute('y', startY - 6);
          const expR = R.get(`${s}|${a}`) || 0;
          lbl.textContent = `${a}: p=${p.toFixed(2)}, r=${expR.toFixed(2)}`;
          svg.appendChild(lbl);
        } else {
          // line with arrow
          const dx = to.x - from.x, dy = to.y - from.y;
          const L = Math.hypot(dx, dy);
          const ux = dx / L, uy = dy / L;
          const x1 = from.x + ux * rNode;
          const y1 = from.y + uy * rNode;
          const x2 = to.x - ux * rNode;
          const y2 = to.y - uy * rNode;
          const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
          line.setAttribute('x1', x1);
          line.setAttribute('y1', y1);
          line.setAttribute('x2', x2);
          line.setAttribute('y2', y2);
          line.setAttribute('class', 'edge');
          line.setAttribute('stroke', color);
          line.setAttribute('stroke-width', String(1 + 4 * p));
          line.setAttribute('marker-end', 'url(#arrow)');
          svg.appendChild(line);
          const lbl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          lbl.setAttribute('class', 'edge-label');
          lbl.setAttribute('x', (x1 + x2) / 2 + 6);
          lbl.setAttribute('y', (y1 + y2) / 2 - 6);
          const expR = R.get(`${s}|${a}`) || 0;
          lbl.textContent = `${a}: p=${p.toFixed(2)}, r=${expR.toFixed(2)}`;
          svg.appendChild(lbl);
        }
      }
    }
  }

  // Draw nodes
  for (const s of states) {
    const i = idx.get(s);
    const gNode = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    gNode.setAttribute('class', 'node');
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    const rNode = 20 + (d[i] ? Math.max(0, d[i]) * 20 : 0);
    circle.setAttribute('cx', pos[i].x);
    circle.setAttribute('cy', pos[i].y);
    circle.setAttribute('r', String(rNode));
    gNode.appendChild(circle);
    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', pos[i].x);
    label.setAttribute('y', pos[i].y + 4);
    label.setAttribute('text-anchor', 'middle');
    label.textContent = String(s);
    gNode.appendChild(label);
    svg.appendChild(gNode);
  }

  container.appendChild(svg);
}

function compute() {
  const errorEl = document.getElementById('error');
  errorEl.style.display = 'none';
  errorEl.textContent = '';
  try {
    const rows = parseCSV(document.getElementById('csvInput').value);
    const model = buildModel(rows);
    const criterion = document.getElementById('criterion')?.value || 'discounted';
    if (criterion === 'discounted') {
      const gamma = Number(document.getElementById('gamma')?.value || 0.95);
      const theta = Number(document.getElementById('theta')?.value || 1e-8);
      const maxIter = Number(document.getElementById('maxIter')?.value || 10000);
      const { V, policy, iterations } = valueIteration(model, gamma, theta, maxIter);
      const states = model.states;
      document.getElementById('iterations').textContent = String(iterations);
      document.getElementById('value').innerHTML = formatVector(states, V);
      document.getElementById('policy').innerHTML = formatPolicy(states, policy);
      setResultsVisibility({ discounted: true });
      // Save and render visualization (no stationary dist; use uniform)
      currentModel = model;
      currentResult = { policy, d: Array(states.length).fill(0) };
      populateActionFilter(getUniqueActions(model));
      if (!vizListenersAttached) {
        document.getElementById('actionFilter').addEventListener('change', renderMDPGraph);
        document.getElementById('policyOnly').addEventListener('change', renderMDPGraph);
        vizListenersAttached = true;
      }
      renderMDPGraph();
    } else {
      const result = policyIteration(model);
      const { policy, d, g, h } = result;
      const states = model.states;
      const V = Array(states.length).fill(g);
      document.getElementById('gain').textContent = g.toFixed(6);
      document.getElementById('value').innerHTML = formatVector(states, V);
      document.getElementById('bias').innerHTML = formatVector(states, h);
      document.getElementById('dist').innerHTML = formatVector(states, d);
      document.getElementById('policy').innerHTML = formatPolicy(states, policy);
      setResultsVisibility({ discounted: false });
      currentModel = model;
      currentResult = result;
      populateActionFilter(getUniqueActions(model));
      if (!vizListenersAttached) {
        document.getElementById('actionFilter').addEventListener('change', renderMDPGraph);
        document.getElementById('policyOnly').addEventListener('change', renderMDPGraph);
        vizListenersAttached = true;
      }
      renderMDPGraph();
    }
  } catch (e) {
    errorEl.style.display = '';
    errorEl.textContent = e.message || String(e);
  }
}

document.getElementById('loadSample').addEventListener('click', loadSample);
document.getElementById('compute').addEventListener('click', compute);

// Auto-load sample for convenience
loadSample();

function setResultsVisibility({ discounted }) {
  const show = (id, visible) => {
    const el = document.getElementById(id);
    if (el) el.style.display = visible ? '' : 'none';
  };
  show('iterGroup', discounted);
  show('gainGroup', !discounted);
  show('biasGroup', !discounted);
  show('distGroup', !discounted);
  show('valueGroup', true);
  show('policyGroup', true);
}