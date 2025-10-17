"""
Outils de programmation linéaire:
- Jeu matriciel (zéro-somme): stratégies mixtes et valeur du jeu.
- MDP gain moyen: politique stationnaire optimale via mesures d'occupation (LP).

Usage jeu matriciel:
  python PL/main.py
  python PL/main.py --csv chemin_vers_matrice.csv

Usage MDP gain moyen:
  python PL/main.py --mdp-sample
  python PL/main.py --mdp-csv chemin_vers_transitions.csv

Format CSV MDP (entêtes facultatives):
  state,action,next_state,prob,reward
Chaque (state,action) doit avoir des probas qui somment à 1.
"""

import sys
import csv
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import pulp as plp

# ======================
# Jeu matriciel (zéro-somme)
# ======================

def read_csv_matrix(path: str) -> List[List[float]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [[float(x) for x in row] for row in csv.reader(f) if row]


def solve_zero_sum_game(A: List[List[float]]) -> Tuple[List[float], List[float], float]:
    m = len(A)
    if m == 0:
        raise ValueError("La matrice ne doit pas être vide.")
    n = len(A[0])
    if any(len(row) != n for row in A):
        raise ValueError("Toutes les lignes doivent avoir la même longueur.")

    # Primal (Joueur 1): max v s.t. A^T y >= v, sum(y)=1, y>=0
    y = [plp.LpVariable(f"y_{i}", lowBound=0) for i in range(m)]
    v = plp.LpVariable("v")
    primal = plp.LpProblem("primal_matrix_game", plp.LpMaximize)
    primal += v
    for j in range(n):
        primal += plp.lpSum(A[i][j] * y[i] for i in range(m)) >= v
    primal += plp.lpSum(y) == 1
    primal.solve(plp.PULP_CBC_CMD(msg=False))
    y_sol = [float(plp.value(var) or 0.0) for var in y]
    v_sol = float(plp.value(v))

    # Dual (Joueur 2): min u s.t. A x <= u, sum(x)=1, x>=0
    x = [plp.LpVariable(f"x_{j}", lowBound=0) for j in range(n)]
    u = plp.LpVariable("u")
    dual = plp.LpProblem("dual_matrix_game", plp.LpMinimize)
    dual += u
    for i in range(m):
        dual += plp.lpSum(A[i][j] * x[j] for j in range(n)) <= u
    dual += plp.lpSum(x) == 1
    dual.solve(plp.PULP_CBC_CMD(msg=False))
    x_sol = [float(plp.value(var) or 0.0) for var in x]

    return y_sol, x_sol, v_sol

# ======================
# MDP gain moyen (LP mesures d'occupation)
# ======================

Transition = Tuple[str, str, str, float, float]  # (s, a, s', p, r)


def read_mdp_csv(path: str) -> List[Transition]:
    rows: List[Transition] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for raw in reader:
            if not raw or raw[0].strip().lower() == "state":
                # ignore entête
                continue
            s, a, sp, p, r = raw[0].strip(), raw[1].strip(), raw[2].strip(), float(raw[3]), float(raw[4])
            rows.append((s, a, sp, p, r))
    return rows


def solve_average_reward_mdp(transitions: List[Transition]) -> Tuple[Dict[str, Dict[str, float]], float, Dict[str, float]]:
    if not transitions:
        raise ValueError("Transitions MDP vides.")

    states = sorted({t[0] for t in transitions} | {t[2] for t in transitions})
    actions_by_state: Dict[str, List[str]] = defaultdict(list)
    prob_sum: Dict[Tuple[str, str], float] = defaultdict(float)
    expected_reward: Dict[Tuple[str, str], float] = defaultdict(float)
    P: Dict[Tuple[str, str, str], float] = {}

    for s, a, sp, p, r in transitions:
        if a not in actions_by_state[s]:
            actions_by_state[s].append(a)
        prob_sum[(s, a)] += p
        expected_reward[(s, a)] += p * r
        P[(s, a, sp)] = P.get((s, a, sp), 0.0) + p

    # Validation proba par (s,a)
    for (s, a), total in prob_sum.items():
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Les probabilités pour ({s},{a}) ne somment pas à 1: {total}")
    # Variables d'occupation x(s,a)
    x_vars: Dict[Tuple[str, str], plp.LpVariable] = {
        (s, a): plp.LpVariable(f"x_{s}_{a}", lowBound=0) for s in actions_by_state for a in actions_by_state[s]
    }

    model = plp.LpProblem("mdp_average_reward", plp.LpMaximize)
    # Objectif: somme x(s,a)*E[r|s,a]
    model += plp.lpSum(x_vars[(s, a)] * expected_reward[(s, a)] for s in actions_by_state for a in actions_by_state[s])
    # Normalisation
    model += plp.lpSum(x_vars.values()) == 1
    # Conservation de flux: pour chaque état s', occupation = flux entrant
    for sp in states:
        lhs = plp.lpSum(x_vars[(s, a)] * P.get((s, a, sp), 0.0) for s in actions_by_state for a in actions_by_state[s])
        rhs = plp.lpSum(x_vars[(sp, a)] for a in actions_by_state.get(sp, []))
        model += lhs == rhs

    model.solve(plp.PULP_CBC_CMD(msg=False))
    if plp.LpStatus[model.status] != "Optimal":
        raise RuntimeError(f"Résolution MDP non optimale: {plp.LpStatus[model.status]}")

    # Politique stationnaire pi(a|s) = x(s,a) / d(s) où d(s)=sum_a x(s,a)
    d: Dict[str, float] = {s: 0.0 for s in states}
    for s in states:
        d[s] = sum(float(plp.value(x_vars[(s, a)]) or 0.0) for a in actions_by_state.get(s, []))
    policy: Dict[str, Dict[str, float]] = {s: {} for s in states}
    for s in states:
        denom = d[s]
        for a in actions_by_state.get(s, []):
            xsa = float(plp.value(x_vars[(s, a)]) or 0.0)
            policy[s][a] = (xsa / denom) if denom > 0 else 0.0
    g_star = float(plp.value(model.objective))
    return policy, g_star, d


def build_sample_mdp_from_figure() -> List[Transition]:
    """
    Exemple inspiré de votre schéma (2 états, 2 actions):
    - État 1, action a: 1/2 reste en 1 (r=1), 1/2 va en 2 (r=1)
    - État 1, action b: va en 2 (r=2) avec proba 1
    - État 2, action a: 1/2 reste en 2 (r=0), 1/2 va en 1 (r=0)
    - État 2, action b: reste en 2 (r=3) avec proba 1
    Adaptez ces valeurs si votre figure diffère.
    """
    rows: List[Transition] = []
    rows += [("1", "a", "1", 0.5, 1.0), ("1", "a", "2", 0.5, 1.0)]
    rows += [("1", "b", "2", 1.0, 2.0)]
    rows += [("2", "a", "2", 0.5, 0.0), ("2", "a", "1", 0.5, 0.0)]
    rows += [("2", "b", "2", 1.0, 3.0)]
    return rows

# ======================
# CLI
# ======================

def main(args: List[str]) -> None:
    if len(args) >= 1 and args[0] == "--mdp-csv":
        mdp_rows = read_mdp_csv(args[1])
        policy, g_star, d = solve_average_reward_mdp(mdp_rows)
        print("MDP (depuis CSV):")
        print(f"États: {sorted({t[0] for t in mdp_rows} | {t[2] for t in mdp_rows})}")
        print("Politique optimale stationnaire (pi(a|s)):")
        for s in sorted(policy.keys()):
            print(f"  état {s}: {policy[s]}")
        print(f"Gain moyen optimal g*: {g_star:.6f}")
        print("Distribution stationnaire d(s):")
        for s in sorted(d.keys()):
            print(f"  d({s}) = {d[s]:.6f}")
        return

    if len(args) >= 1 and args[0] == "--mdp-sample":
        mdp_rows = build_sample_mdp_from_figure()
        policy, g_star, d = solve_average_reward_mdp(mdp_rows)
        print("MDP (exemple inspiré du schéma):")
        print("Transitions:")
        for row in mdp_rows:
            print(row)
        print("Politique optimale stationnaire (pi(a|s)):")
        for s in sorted(policy.keys()):
            print(f"  état {s}: {policy[s]}")
        print(f"Gain moyen optimal g*: {g_star:.6f}")
        print("Distribution stationnaire d(s):")
        for s in sorted(d.keys()):
            print(f"  d({s}) = {d[s]:.6f}")
        return

    # Par défaut: jeu matriciel d'exemple
    A = [
        [2, -1, 3],
        [0,  1, -2],
        [1, -1, 0],
    ]
    y, x, v = solve_zero_sum_game(A)

    print("Matrice de gains (Joueur 1):")
    for row in A:
        print(row)

    print(f"\nStratégie optimale joueur 1 (lignes): {y}")
    print(f"Stratégie optimale joueur 2 (colonnes): {x}")
    print(f"Valeur du jeu (gain moyen garanti): {v:.6f}")

    m = len(A)
    n = len(A[0])
    exp_cols = [sum(A[i][j] * y[i] for i in range(m)) for j in range(n)]
    exp_rows = [sum(A[i][j] * x[j] for j in range(n)) for i in range(m)]
    print(f"Min des colonnes (>= valeur): {min(exp_cols):.6f}")
    print(f"Max des lignes (<= valeur): {max(exp_rows):.6f}")


if __name__ == "__main__":
    main(sys.argv[1:])