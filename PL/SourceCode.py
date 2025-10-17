"""
Recherche de stratégie à gain moyen optimal (MDP non escompté)
Politique itérative (amélioration de Bellman) avec sorties détaillées par étape.

Format CSV accepté (EN/FR):
- state,action,next_state,prob,reward
- etat,action,etat_suivant,proba,gain

Ce script:
1) Parse le CSV et construit le modèle (états, actions, P, R).
2) Évalue une politique π: calcule Pπ, rπ, la distribution stationnaire d, le gain moyen g,
   puis résout (I - Pπ) h = rπ - g·1 avec la contrainte h(s0)=0 pour obtenir le biais h.
3) Améliore la politique par Q(s,a) = E[r|s,a] - g + Σ_j p(s,a,j) h(j), choisit l'action maximisant Q.
4) Répète jusqu'à stabilisation et affiche toutes les matrices/vecteurs à chaque itération.

Exécution:
- python woho.py                # utilise un exemple intégré
- python woho.py --csv chemin.csv
"""
from typing import Dict, List, Tuple
import argparse

Transition = Tuple[float, str, float]  # (prob, next_state, reward)
TransitionTable = Dict[Tuple[str, str], List[Transition]]

# ---------- Parsing & Modèle ----------
def parse_csv_text(text: str) -> List[Dict[str, str]]:
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split(',')]
        if len(cells) < 5:
            continue
        first = cells[0].lower()
        if first in ('state', 'etat'):
            # Header
            continue
        try:
            p = float(cells[3]); r = float(cells[4])
        except ValueError:
            continue
        rows.append({'s': cells[0], 'a': cells[1], 'sp': cells[2], 'p': p, 'r': r})
    return rows

def build_model(rows: List[Dict[str, str]]):
    states_set = set()
    actions_by_state: Dict[str, List[str]] = {}
    P: Dict[Tuple[str, str, str], float] = {}
    R: Dict[Tuple[str, str], float] = {}
    PSum: Dict[Tuple[str, str], float] = {}

    for row in rows:
        s, a, sp, p, r = row['s'], row['a'], row['sp'], float(row['p']), float(row['r'])
        states_set.update([s, sp])
        actions_by_state.setdefault(s, [])
        if a not in actions_by_state[s]:
            actions_by_state[s].append(a)
        P[(s, a, sp)] = P.get((s, a, sp), 0.0) + p
        R[(s, a)] = R.get((s, a), 0.0) + p * r
        PSum[(s, a)] = PSum.get((s, a), 0.0) + p

    states = sorted(states_set)
    # Valider que Σ_{sp} p(s,a,sp) = 1
    for (s, a), total in PSum.items():
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Les probabilités pour ({s},{a}) ne somment pas à 1: {total}")

    # Trier les actions par état
    for s in actions_by_state:
        actions_by_state[s].sort()

    return {
        'states': states,
        'actions_by_state': actions_by_state,
        'P': P,
        'R': R,
    }

# ---------- Outils algèbre linéaire ----------
def solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    """Élimination de Gauss avec pivot partiel."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r
        if abs(M[pivot][col]) < 1e-12:
            continue
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
        pv = M[col][col]
        for c in range(col, n + 1):
            M[col][c] /= pv
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if abs(factor) < 1e-12:
                continue
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    return [M[i][n] for i in range(n)]

def stationary_distribution(Ppi: List[List[float]]) -> List[float]:
    """Résout (I - P^T) d = 0 avec contrainte Σ d = 1."""
    n = len(Ppi)
    # Construire A = I - P^T
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for i in range(n):
        for j in range(n):
            A[i][j] = (1.0 if i == j else 0.0) - Ppi[j][i]
    # Remplacer la dernière équation par Σ d = 1
    for j in range(n):
        A[n - 1][j] = 1.0
    b[n - 1] = 1.0
    return solve_linear(A, b)

# ---------- Évaluation de politique (gain moyen) ----------
def evaluate_policy_gain(model, policy):
    states: List[str] = model['states']
    P = model['P']; R = model['R']
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    Ppi = [[0.0] * n for _ in range(n)]
    rpi = [0.0] * n

    for s in states:
        a = policy.get(s)
        i = idx[s]
        rpi[i] = R.get((s, a), 0.0)
        for sp in states:
            p = P.get((s, a, sp), 0.0)
            j = idx[sp]
            Ppi[i][j] += p

    d = stationary_distribution(Ppi)
    g = sum(d[i] * rpi[i] for i in range(n))

    # Résoudre (I - Pπ) h = rπ - g 1 avec la contrainte h(s0)=0
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for i in range(n):
        for j in range(n):
            A[i][j] = (1.0 if i == j else 0.0) - Ppi[i][j]
        b[i] = rpi[i] - g
    # Contraindre h(s0)=0 (remplacer la première ligne)
    for j in range(n):
        A[0][j] = 1.0 if j == 0 else 0.0
    b[0] = 0.0
    h = solve_linear(A, b)

    return {
        'states': states,
        'Ppi': Ppi,
        'rpi': rpi,
        'd': d,
        'g': g,
        'A': A,
        'b': b,
        'h': h,
    }

# ---------- Amélioration de politique ----------
def q_value(model, h_vec: List[float], g: float, s: str, a: str) -> float:
    states = model['states']; P = model['P']; R = model['R']
    idx = {st: i for i, st in enumerate(states)}
    expR = R.get((s, a), 0.0)
    sum_next = 0.0
    for sp in states:
        p = P.get((s, a, sp), 0.0)
        j = idx[sp]
        sum_next += p * h_vec[j]
    return expR - g + sum_next

def greedy_improve(model, eval):
    states = model['states']; actions_by_state = model['actions_by_state']
    h = eval['h']; g = eval['g']
    new_policy: Dict[str, str] = {}
    Qlog = []
    for s in states:
        acts = actions_by_state.get(s, [])
        best_a = acts[0]
        best_q = float('-inf')
        per_state = []
        for a in acts:
            q = q_value(model, h, g, s, a)
            per_state.append((a, q))
            if q > best_q + 1e-12:
                best_q, best_a = q, a
        new_policy[s] = best_a
        Qlog.append({'s': s, 'qs': per_state})
    return new_policy, Qlog

def policies_equal(p1: Dict[str, str], p2: Dict[str, str]) -> bool:
    return p1.keys() == p2.keys() and all(p1[s] == p2[s] for s in p1)

def initial_policy(model) -> Dict[str, str]:
    states = model['states']; actions_by_state = model['actions_by_state']; R = model['R']
    pol: Dict[str, str] = {}
    for s in states:
        acts = actions_by_state.get(s, [])
        best_a = acts[0]
        best_r = float('-inf')
        for a in acts:
            expR = R.get((s, a), 0.0)
            if expR > best_r + 1e-12:
                best_r, best_a = expR, a
        pol[s] = best_a
    return pol

# ---------- Formatage pour affichage ----------
def fmt_vec(states: List[str], vec: List[float], name: str) -> str:
    return name + ': ' + ', '.join(f"{s}:{vec[i]:.6f}" for i, s in enumerate(states))

def fmt_mat(states: List[str], M: List[List[float]], name: str) -> str:
    lines = [name]
    header = '      ' + '  '.join(states)
    lines.append(header)
    for i, s in enumerate(states):
        row = '[' + ', '.join(f"{M[i][j]:.6f}" for j in range(len(states))) + ']'
        lines.append(f"{s} {row}")
    return '\n'.join(lines)

# ---------- Boucle principale (verbose) ----------
def policy_iteration_gain_verbose(model, max_iter: int = 50):
    policy = initial_policy(model)
    print(f"États: {model['states']}")
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        print(f"\n== Itération d'amélioration {iterations}")
        print('Politique courante π:')
        for s in model['states']:
            print(f"  π({s}) = {policy.get(s)}")
        eval = evaluate_policy_gain(model, policy)
        states, Ppi, rpi, d, g, A, b, h = (
            eval['states'], eval['Ppi'], eval['rpi'], eval['d'], eval['g'], eval['A'], eval['b'], eval['h']
        )
        print('\n' + fmt_mat(states, Ppi, 'Pπ'))
        print(fmt_vec(states, rpi, 'rπ'))
        print(fmt_vec(states, d,   'd (stationnaire)'))
        print(f"g (gain moyen): {g:.6f}")
        print('\n' + fmt_mat(states, A, 'A = I - Pπ'))
        print(fmt_vec(states, b, 'b = rπ - g·1'))
        print(fmt_vec(states, h, 'h (solution de A h = b, avec h(s0)=0)'))
        new_policy, Qlog = greedy_improve(model, eval)
        print('\nQ-valeurs et amélioration gloutonne:')
        for entry in Qlog:
            s = entry['s']
            qs_txt = ', '.join(f"Q({s},{a})={q:.6f}" for a, q in entry['qs'])
            print(f"  {qs_txt} ⇒ choisir {new_policy[s]}")
        if policies_equal(new_policy, policy):
            print('\nPolitique stabilisée. Arrêt.')
            return eval, policy, iterations
        policy = new_policy
    return eval, policy, iterations

# ---------- Exemple & CLI ----------
def example_csv() -> str:
    return '\n'.join([
        'etat,action,etat_suivant,proba,gain',
        '1,a,1,0.5,1',
        '1,a,2,0.5,1',
        '1,b,2,1.0,2',
        '2,a,2,0.5,0',
        '2,a,1,0.5,0',
        '2,b,2,1.0,3',
    ])

def main():
    parser = argparse.ArgumentParser(description='Recherche de stratégie à gain moyen optimal (MDP non escompté)')
    parser.add_argument('--csv', type=str, help='Chemin vers un fichier CSV (state,action,next_state,prob,reward)')
    args = parser.parse_args()

    if args.csv:
        with open(args.csv, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = example_csv()

    rows = parse_csv_text(text)
    model = build_model(rows)
    eval, policy, iters = policy_iteration_gain_verbose(model)

    print('\n== Résumé ==')
    print(f'Convergence après {iters} itérations d\'amélioration')
    print(f"g*: {eval['g']:.6f}")
    print(fmt_vec(eval['states'], eval['h'], 'h* (biais)'))
    print(fmt_vec(eval['states'], [eval['g']] * len(eval['states']), 'V(s) (gain moyen par état)'))
    print('Politique optimale π*:')
    for s in eval['states']:
        print(f"  {s}: {policy[s]}")

if __name__ == '__main__':
    main()