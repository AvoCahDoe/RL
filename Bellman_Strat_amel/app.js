// Bellman Strategy Improvement (Discounted Policy Iteration) — Verbose
// CSV format: state,action,next_state,prob,reward

function parseCSV(text) {
  const rows = [];
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  for (const line of lines) {
    const cells = line.split(',').map(c => c.trim());
    if (cells.length < 5) continue;
    if (isNaN(Number(cells[3])) || isNaN(Number(cells[4]))) continue;
    const first = cells[0].toLowerCase();
    if (first === 'state' || first === 'etat') continue; // skip header EN/FR
    rows.push({ s: cells[0], a: cells[1], sp: cells[2], p: Number(cells[3]), r: Number(cells[4]) });
  }
  return rows;
}

function buildModel(rows) {
  const statesSet = new Set();
  const actionsByState = new Map();
  const P = new Map(); // key `${s}|${a}|${sp}` -> prob
  const R = new Map(); // key `${s}|${a}` -> expected reward
  const PSum = new Map(); // prob sum per (s,a)
  for (const { s, a, sp, p, r } of rows) {
    statesSet.add(s); statesSet.add(sp);
    if (!actionsByState.has(s)) actionsByState.set(s, new Set());
    actionsByState.get(s).add(a);
    const key = `${s}|${a}|${sp}`;
    P.set(key, (P.get(key) || 0) + p);
    const raKey = `${s}|${a}`;
    R.set(raKey, (R.get(raKey) || 0) + p * r);
    PSum.set(raKey, (PSum.get(raKey) || 0) + p);
  }
  const states = Array.from(statesSet).sort();
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
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < 1e-12) continue;
    if (pivot !== col) { const tmp = M[col]; M[col] = M[pivot]; M[pivot] = tmp; }
    const pv = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= pv;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      if (Math.abs(factor) < 1e-12) continue;
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map(row => row[n]);
}

function evaluatePolicyDetailed(model, policy, gamma) {
  const { states, P, R } = model;
  const n = states.length;
  const idx = new Map(states.map((s, i) => [s, i]));
  const Ppi = makeMatrix(n, 0);
  const Rpi = Array(n).fill(0);
  for (const s of states) {
    const a = policy.get(s);
    const i = idx.get(s);
    const raKey = `${s}|${a}`;
    Rpi[i] = R.get(raKey) || 0;
    for (const sp of states) {
      const key = `${s}|${a}|${sp}`;
      const p = P.get(key) || 0;
      const j = idx.get(sp);
      Ppi[i][j] += p;
    }
  }
  const A = makeMatrix(n, 0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      A[i][j] = (i === j ? 1 : 0) - gamma * Ppi[i][j];
    }
  }
  const Vvec = solveLinear(A, Rpi);
  const V = new Map(states.map((s, i) => [s, Vvec[i]]));
  return { states, Ppi, Rpi, A, V };
}

function qValue(model, V, s, a, gamma) {
  const { states, P, R } = model;
  let q = 0;
  for (const sp of states) {
    const p = P.get(`${s}|${a}|${sp}`) || 0;
    const r = (R.get(`${s}|${a}`) || 0); // expected reward already includes prob weighting
    q += p * (r + gamma * (V.get(sp) || 0));
  }
  return q;
}

function greedyImprove(model, V, gamma) {
  const { states, actionsByState } = model;
  const newPolicy = new Map();
  const Qlog = [];
  for (const s of states) {
    const actions = Array.from(actionsByState.get(s) || []);
    let bestA = actions[0];
    let bestQ = -Infinity;
    const qs = [];
    for (const a of actions) {
      const q = qValue(model, V, s, a, gamma);
      qs.push({ a, q });
      if (q > bestQ + 1e-12) { bestQ = q; bestA = a; }
    }
    newPolicy.set(s, bestA);
    Qlog.push({ s, qs });
  }
  return { newPolicy, Qlog };
}

function policiesEqual(p1, p2) {
  if (p1.size !== p2.size) return false;
  for (const [s, a] of p1.entries()) {
    if (p2.get(s) !== a) return false;
  }
  return true;
}

function initialPolicy(model) {
  const { states, actionsByState, R } = model;
  const pol = new Map();
  for (const s of states) {
    const actions = Array.from(actionsByState.get(s) || []);
    let bestA = actions[0];
    let bestR = -Infinity;
    for (const a of actions) {
      const expR = R.get(`${s}|${a}`) || 0;
      if (expR > bestR + 1e-12) { bestR = expR; bestA = a; }
    }
    pol.set(s, bestA);
  }
  return pol;
}

function policyIterationVerbose(model, gamma, maxIter) {
  const iterations = [];
  let policy = initialPolicy(model);
  for (let iter = 1; iter <= maxIter; iter++) {
    const eval = evaluatePolicyDetailed(model, policy, gamma);
    const { newPolicy, Qlog } = greedyImprove(model, eval.V, gamma);
    iterations.push({ iter, policy: new Map(policy), ...eval, Qlog, newPolicy });
    if (policiesEqual(newPolicy, policy)) break;
    policy = newPolicy;
  }
  const last = iterations[iterations.length - 1];
  const finalV = last.V;
  const finalPolicy = last.newPolicy || last.policy;
  return { iterations, finalV, finalPolicy };
}

function formatVector(states, vec) {
  let html = '<table class="table"><thead><tr><th>State</th><th>Value</th></tr></thead><tbody>';
  states.forEach((s, i) => { html += `<tr><td>${s}</td><td>${Number(vec[i].toFixed(6))}</td></tr>`; });
  html += '</tbody></table>';
  return html;
}

function formatPolicy(states, policy) {
  let html = '<table class="table"><thead><tr><th>State</th><th>Action</th></tr></thead><tbody>';
  states.forEach(s => { html += `<tr><td>${s}</td><td>${policy.get(s)}</td></tr>`; });
  html += '</tbody></table>';
  return html;
}

function formatMatrix(states, M, title) {
  let html = `<div class="matrix"><h4>${title}</h4><table class="table"><thead><tr><th></th>`;
  states.forEach(s => { html += `<th>${s}</th>`; });
  html += '</tr></thead><tbody>';
  for (let i = 0; i < states.length; i++) {
    html += `<tr><td>${states[i]}</td>`;
    for (let j = 0; j < states.length; j++) {
      html += `<td>${Number(M[i][j].toFixed(6))}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table></div>';
  return html;
}

function formatQlog(Qlog) {
  let html = '<div class="qlog"><h4>Q-values</h4><table class="table"><thead><tr><th>State</th><th>Action</th><th>Q(s,a)</th></tr></thead><tbody>';
  for (const { s, qs } of Qlog) {
    for (const { a, q } of qs) {
      html += `<tr><td>${s}</td><td>${a}</td><td>${Number(q.toFixed(6))}</td></tr>`;
    }
  }
  html += '</tbody></table></div>';
  return html;
}

function iterationBlockHTML(detail) {
  const { iter, states, policy, Ppi, Rpi, A, V, Qlog, newPolicy } = detail;
  let html = `<div class="iteration-block"><h3>Iteration ${iter}</h3>`;
  html += '<h4>Current policy π</h4>' + formatPolicy(states, policy);
  html += formatMatrix(states, Ppi, 'Pπ');
  html += '<div class="vector"><h4>Rπ</h4>' + formatVector(states, Rpi) + '</div>';
  html += formatMatrix(states, A, 'A = I - γ Pπ');
  html += '<div class="vector"><h4>V from A V = Rπ</h4>' + formatVector(states, states.map(s => V.get(s))) + '</div>';
  html += formatQlog(Qlog);
  html += '<h4>Improved policy π′</h4>' + formatPolicy(states, newPolicy);
  html += '</div>';
  return html;
}

function loadSample() {
  const sample = [
    'state,action,next_state,prob,reward',
    '1,a,1,0.5,2',
    '1,a,2,0.5,1',
    '1,b,2,1.0,0.2',
    '2,a,2,0.5,0',
    '2,a,1,0.5,0',
    '2,b,2,1.0,0.3',
  ].join('\n');
  document.getElementById('csvInput').value = sample;
}

function compute() {
  const errorEl = document.getElementById('error');
  errorEl.style.display = 'none';
  errorEl.textContent = '';
  try {
    const rows = parseCSV(document.getElementById('csvInput').value);
    const gamma = Number(document.getElementById('gammaInput').value);
    const maxIter = Number(document.getElementById('maxIterInput').value);
    if (!(gamma > 0 && gamma < 1)) throw new Error('Gamma must be in (0,1).');
    if (!(maxIter >= 1)) throw new Error('Max iterations must be >= 1.');
    const model = buildModel(rows);
    const { iterations, finalV, finalPolicy } = policyIterationVerbose(model, gamma, maxIter);
    const iterEl = document.getElementById('iterations');
    iterEl.innerHTML = iterations.map(iter => iterationBlockHTML(iter)).join('');
    const states = model.states;
    const Vvec = states.map(s => finalV.get(s));
    document.getElementById('value').innerHTML = formatVector(states, Vvec);
    document.getElementById('policy').innerHTML = formatPolicy(states, finalPolicy);
    document.getElementById('iterCount').textContent = String(iterations.length);
    document.getElementById('gammaOut').textContent = String(gamma);
  } catch (e) {
    errorEl.style.display = '';
    errorEl.textContent = e.message || String(e);
  }
}

// Bind events
document.getElementById('loadSample').addEventListener('click', loadSample);
document.getElementById('compute').addEventListener('click', compute);

// Auto-load sample
loadSample();