"""
Bellman Strategy Improvement (Policy Iteration) for discounted MDPs.
Verbose run prints every phase: states, Pπ, Rπ, A=(I-γPπ), V, Q(s,a), and policy updates.
Default gamma = 0.5 (α = 1/2).
"""
from typing import Dict, List, Tuple

TransitionTable = Dict[Tuple[str, str], List[Tuple[float, str, float]]]


def states_from(transitions: TransitionTable) -> List[str]:
    s: set[str] = set()
    for (st, _), outcomes in transitions.items():
        s.add(st)
        for p, sp, r in outcomes:
            s.add(sp)
    return sorted(s)


def build_actions(transitions: TransitionTable) -> Dict[str, List[str]]:
    by_state: Dict[str, List[str]] = {}
    for (st, a) in transitions:
        by_state.setdefault(st, [])
        if a not in by_state[st]:
            by_state[st].append(a)
    for st in by_state:
        by_state[st].sort()
    return by_state


def solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
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


def evaluate_policy(transitions: TransitionTable, policy: Dict[str, str], gamma: float) -> Dict[str, float]:
    states = states_from(transitions)
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    Ppi = [[0.0 for _ in range(n)] for _ in range(n)]
    Rpi = [0.0 for _ in range(n)]
    for s in states:
        a = policy.get(s)
        if a is None:
            continue
        i = idx[s]
        outcomes = transitions.get((s, a), [])
        prob_sum = 0.0
        for p, sp, r in outcomes:
            j = idx[sp]
            Ppi[i][j] += p
            Rpi[i] += p * r
            prob_sum += p
        if abs(prob_sum - 1.0) > 1e-8:
            raise ValueError(f"Probabilities for ({s},{a}) do not sum to 1: {prob_sum}")
    A = [[(1.0 if i == j else 0.0) - gamma * Ppi[i][j] for j in range(n)] for i in range(n)]
    V_vec = solve_linear(A, Rpi)
    return {s: V_vec[idx[s]] for s in states}


def q_value(transitions: TransitionTable, V: Dict[str, float], s: str, a: str, gamma: float) -> float:
    q = 0.0
    for p, sp, r in transitions.get((s, a), []):
        q += p * (r + gamma * V.get(sp, 0.0))
    return q


def policies_equal(p1: Dict[str, str], p2: Dict[str, str]) -> bool:
    return p1.keys() == p2.keys() and all(p1[s] == p2[s] for s in p1)


def initial_policy(transitions: TransitionTable) -> Dict[str, str]:
    actions = build_actions(transitions)
    pol: Dict[str, str] = {}
    for s, acts in actions.items():
        best_a, best_r = acts[0], float('-inf')
        for a in acts:
            exp_r = sum(p * r for p, sp, r in transitions[(s, a)])
            if exp_r > best_r + 1e-12:
                best_r, best_a = exp_r, a
        pol[s] = best_a
    return pol


def compute_evaluation_details(transitions: TransitionTable, policy: Dict[str, str], gamma: float):
    states = states_from(transitions)
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    Ppi = [[0.0 for _ in range(n)] for _ in range(n)]
    Rpi = [0.0 for _ in range(n)]
    for s in states:
        a = policy.get(s)
        if a is None:
            continue
        i = idx[s]
        outcomes = transitions.get((s, a), [])
        prob_sum = 0.0
        for p, sp, r in outcomes:
            j = idx[sp]
            Ppi[i][j] += p
            Rpi[i] += p * r
            prob_sum += p
        if abs(prob_sum - 1.0) > 1e-8:
            raise ValueError(f"Probabilities for ({s},{a}) do not sum to 1: {prob_sum}")
    A = [[(1.0 if i == j else 0.0) - gamma * Ppi[i][j] for j in range(n)] for i in range(n)]
    V_vec = solve_linear(A, Rpi)
    V = {s: V_vec[idx[s]] for s in states}
    return states, Ppi, Rpi, A, V


def fmt_vector(vec: List[float], labels: List[str] | None = None, decimals: int = 6) -> str:
    def f(x: float) -> str:
        return f"{x:.{decimals}f}"
    if labels:
        return "[" + ", ".join(f"{labels[i]}:{f(vec[i])}" for i in range(len(vec))) + "]"
    return "[" + ", ".join(f(x) for x in vec) + "]"


def fmt_matrix(M: List[List[float]], row_labels: List[str] | None = None, col_labels: List[str] | None = None, decimals: int = 6) -> str:
    def f(x: float) -> str:
        return f"{x:.{decimals}f}"
    rows = []
    if col_labels:
        rows.append("    " + "  ".join(col_labels))
    for i, row in enumerate(M):
        prefix = f"{row_labels[i]} " if row_labels else ""
        rows.append(prefix + "[" + ", ".join(f(x) for x in row) + "]")
    return "\n".join(rows)


def policy_iteration_verbose(transitions: TransitionTable, gamma: float = 0.5, max_iter: int = 100):
    actions = build_actions(transitions)
    policy = initial_policy(transitions)
    print(f"States: {states_from(transitions)}")
    it = 0
    while it < max_iter:
        it += 1
        print(f"\n== Improvement Iteration {it}")
        print("Current policy π:")
        for s in sorted(actions.keys()):
            print(f"  π({s}) = {policy.get(s)}")
        states, Ppi, Rpi, A, V = compute_evaluation_details(transitions, policy, gamma)
        print("\nPπ:")
        print(fmt_matrix(Ppi, row_labels=states, col_labels=states, decimals=6))
        print("Rπ:")
        print(fmt_vector(Rpi, labels=states, decimals=6))
        print("A = I - γ Pπ:")
        print(fmt_matrix(A, row_labels=states, col_labels=states, decimals=6))
        print("Solve (A) V = Rπ ⇒ V:")
        print(fmt_vector([V[s] for s in states], labels=states, decimals=6))
        new_policy: Dict[str, str] = {}
        print("\nQ-values and greedy improvement:")
        for s in states:
            qs: List[Tuple[str, float]] = []
            for a in actions.get(s, []):
                q = q_value(transitions, V, s, a, gamma)
                qs.append((a, q))
            qs.sort(key=lambda t: t[0])
            best_a, best_q = max(qs, key=lambda t: t[1])
            new_policy[s] = best_a
            qs_txt = ", ".join(f"Q({s},{a})={q:.6f}" for a, q in qs)
            print(f"  {qs_txt} ⇒ choose {best_a}")
        if policies_equal(new_policy, policy):
            print("\nPolicy stabilized. Stop.")
            return V, policy, it
        policy = new_policy
    return V, policy, it


# ----- Example MDP (default) -----

def example_mdp() -> TransitionTable:
    t: TransitionTable = {
        ('1', 'a'): [(0.5, '1', 2.0), (0.5, '2', 1.0)],
        ('1', 'b'): [(1.0, '2', 0.2)],
        ('2', 'a'): [(0.5, '2', 0.0), (0.5, '1', 0.0)],
        ('2', 'b'): [(1.0, '2', 0.3)],
    }
    return t


def main():
    transitions = example_mdp()
    gamma = 0.5
    V, policy, iters = policy_iteration_verbose(transitions, gamma=gamma)
    print(f"\n== Summary ==")
    print(f"Converged after {iters} improvement iterations, gamma={gamma}")
    print("Optimal values V*(s):")
    for s in sorted(V):
        print(f"  {s}: {V[s]:.6f}")
    print("Optimal policy π*(s):")
    for s in sorted(policy):
        print(f"  {s}: {policy[s]}")


if __name__ == '__main__':
    main()