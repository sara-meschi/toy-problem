from __future__ import annotations

from typing import Dict, List
import math


def softmax(xs: List[float], beta: float = 1.0) -> List[float]:
    scaled = [beta * x for x in xs]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def move_likelihoods_from_distances(
    next_distances: Dict[int, int],
    beta: float = 2.0,
) -> Dict[int, float]:
    """
    Convert goal-conditioned scores into probabilities.

    next_distances[g] = shortest-path distance from observed next attacker node
                        to candidate goal g.

    Lower distance should mean higher likelihood.
    So we use score = -distance and then softmax.
    """
    goals = list(next_distances.keys())
    scores = [-float(next_distances[g]) for g in goals]
    probs = softmax(scores, beta=beta)
    return {g: p for g, p in zip(goals, probs)}


def bayes_update(
    prior: Dict[int, float],
    likelihood: Dict[int, float],
) -> Dict[int, float]:
    """
    Posterior(goal) ∝ prior(goal) * likelihood(observed_move | goal)
    """
    unnormalized = {}
    for g, p in prior.items():
        unnormalized[g] = p * likelihood[g]

    total = sum(unnormalized.values())
    if total <= 0:
        # fallback to prior if something degenerate happens
        return prior.copy()

    return {g: unnormalized[g] / total for g in prior}


def attacker_move_posterior(
    game,
    prior: Dict[int, float],
    current_attacker_node: int,
    observed_next_attacker_node: int,
    candidate_goals: List[int],
    beta: float = 2.0,
) -> Dict[int, float]:
    """
    Update defender belief after observing attacker move.

    We score the observed next node under each candidate goal
    using shortest-path distance from that next node to the goal.
    """
    if observed_next_attacker_node not in game.adjacency[current_attacker_node]:
        raise ValueError("Observed attacker move is not legal from current node.")

    next_distances = {
        g: game.shortest_distance(observed_next_attacker_node, g)
        for g in candidate_goals
    }

    likelihood = move_likelihoods_from_distances(next_distances, beta=beta)
    posterior = bayes_update(prior, likelihood)
    return posterior