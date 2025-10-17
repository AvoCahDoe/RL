# Value Iteration algorithm for discounted MDP
# Computes V* and the optimal policy from transitions and rewards

from typing import Dict, List, Tuple

TransitionTable = Dict[Tuple[str, str], List[Tuple[float, str, float]]]


def value_iteration(transitions: TransitionTable, gamma: float = 0.95, theta: float = 1e-8, max_iter: int = 10000):
    """
    transitions: dict[(state, action)] -> list of (prob, next_state, reward)
    gamma: discount factor (0 < gamma < 1)
    theta: convergence threshold on the maximum change in V
    max_iter: hard cap on iterations

    Returns: (V, policy, iterations)
    """
    # Collect all states that appear in the transition table
    states = set()
    for (s, a), outcomes in transitions.items():
        states.add(s)
        for p, s_next, r in outcomes:
            states.add(s_next)

    # Initialize state values to zero
    V = {s: 0.0 for s in states}
    iters = 0

    # Value iteration loop (Bellman optimality update)
    while iters < max_iter:
        delta = 0.0
        for s in states:
            actions = [a for (ss, a) in transitions.keys() if ss == s]
            if not actions:
                continue  # state with no available actions
            best_q = float('-inf')
            for a in actions:
                q = 0.0
                for p, s_next, r in transitions[(s, a)]:
                    q += p * (r + gamma * V[s_next])
                if q > best_q:
                    best_q = q
            delta = max(delta, abs(best_q - V[s]))
            V[s] = best_q
        iters += 1
        if delta < theta:
            break

    # Derive the greedy optimal policy with respect to V
    policy = {}
    for s in states:
        actions = [a for (ss, a) in transitions.keys() if ss == s]
        if not actions:
            continue
        best_a, best_q = None, float('-inf')
        for a in actions:
            q = 0.0
            for p, s_next, r in transitions[(s, a)]:
                q += p * (r + gamma * V[s_next])
            if q > best_q:
                best_q, best_a = q, a
        policy[s] = best_a

    return V, policy, iters


def example_mdp() -> TransitionTable:
    """
    Small MDP with three states: s0, s1, s2 (s2 is quasi-terminal).
    Actions: 'go' and 'stay'. Rewards and probabilities defined below.
    """
    return {
        ('s0', 'go'): [(0.8, 's1', 0.0), (0.2, 's0', -0.2)],
        ('s0', 'stay'): [(1.0, 's0', -0.05)],
        ('s1', 'go'): [(0.9, 's2', 1.0), (0.1, 's0', 0.0)],
        ('s1', 'stay'): [(1.0, 's1', -0.05)],
        ('s2', 'stay'): [(1.0, 's2', 0.0)],
        ('s2', 'go'): [(1.0, 's2', 0.0)],
    }


def main():
    transitions = example_mdp()
    gamma = 0.95
    V, policy, iters = value_iteration(transitions, gamma=gamma, theta=1e-8)
    print(f"Converged in {iters} iterations, gamma={gamma}")
    print("Optimal values V*(s):")
    for s in sorted(V):
        print(f"  {s}: {V[s]:.4f}")
    print("Optimal policy pi*(s):")
    for s in sorted(policy):
        print(f"  {s}: {policy[s]}")


if __name__ == "__main__":
    main()