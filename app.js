// MDP Average Gain Solver (Policy Iteration)
// CSV format: state,action,next_state,prob,reward

function parseCSV(text) {
  const rows = [];
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  for (const line of lines) {
    const cells = line.split(',').map(c => c.trim());
    if (cells.length < 5) continue;
    if (isNaN(Number(cells[3])) || isNaN(Number(cells[4]))) continue;
    if (cells[0].toLowerCase() === 'state') continue; // header
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

function formatVector(states, vec, decimals = 6) {
  const obj = {};
  states.forEach((s, i) => { obj[s] = Number(vec[i].toFixed(decimals)); });
  return JSON.stringify(obj, null, 2);
}

function formatPolicy(states, policy) {
  const obj = {};
  states.forEach(s => { obj[s] = policy.get(s); });
  return JSON.stringify(obj, null, 2);
}

function loadSample() {
  const sample = [
    'state,action,next_state,prob,reward',
    '1,a,1,0.5,1',
    '1,a,2,0.5,1',
    '1,b,2,1.0,2',
    '2,a,2,0.5,0',
    '2,a,1,0.5,0',
    '2,b,2,1.0,3',
  ].join('\n');
  document.getElementById('csv').value = sample;
}

function compute() {
  const errorEl = document.getElementById('error');
  errorEl.hidden = true; errorEl.textContent = '';
  try {
    const rows = parseCSV(document.getElementById('csv').value);
    const model = buildModel(rows);
    const result = policyIteration(model);
    const { policy, d, g, h } = result;
    const states = model.states;
    // Gain vector V(s): equal to g for each state in communicating classes
    const V = Array(states.length).fill(g);
    document.getElementById('gstar').textContent = g.toFixed(6);
    document.getElementById('gainVector').textContent = formatVector(states, V);
    document.getElementById('biasVector').textContent = formatVector(states, h);
    document.getElementById('stationary').textContent = formatVector(states, d);
    document.getElementById('policy').textContent = formatPolicy(states, policy);
  } catch (e) {
    errorEl.hidden = false;
    errorEl.textContent = e.message || String(e);
  }
}

document.getElementById('loadSample').addEventListener('click', loadSample);
document.getElementById('compute').addEventListener('click', compute);

// Auto-load sample for convenience
loadSample();