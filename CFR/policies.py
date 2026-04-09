# from __future__ import annotations

# from typing import Dict, List


# def shortest_path_next_move(game, current_node: int, target_node: int) -> int:
#     """
#     Return one neighbor of current_node that minimizes distance to target_node.
#     If there are multiple, return the first one.
#     """
#     neighbors = game.adjacency[current_node]
#     best_nbr = None
#     best_dist = float("inf")

#     for nbr in neighbors:
#         d = game.shortest_distance(nbr, target_node)
#         if d < best_dist:
#             best_dist = d
#             best_nbr = nbr

#     if best_nbr is None:
#         raise ValueError(f"No legal move from node {current_node}")

#     return best_nbr


# def attacker_shortest_path_policy(game, state) -> int:
#     """
#     Attacker moves one step along a shortest path to the true goal.
#     """
#     return shortest_path_next_move(game, state.attacker, state.true_goal)


# def defender_chase_belief_policy(game, state, belief: Dict[int, float]) -> int:
#     """
#     Defender chooses the most likely goal under current belief
#     and moves one step along a shortest path toward it.
#     """
#     most_likely_goal = max(belief, key=belief.get)
#     return shortest_path_next_move(game, state.defender, most_likely_goal)

from __future__ import annotations

from typing import Dict, List, Optional


def shortest_path_next_move(game, current_node: int, target_node: int) -> int:
    neighbors = game.adjacency[current_node]
    best_nbr = None
    best_dist = float("inf")

    for nbr in neighbors:
        d = game.shortest_distance(nbr, target_node)
        if d < best_dist:
            best_dist = d
            best_nbr = nbr

    if best_nbr is None:
        raise ValueError(f"No legal move from node {current_node}")

    return best_nbr


def attacker_shortest_path_policy(game, state) -> int:
    return shortest_path_next_move(game, state.attacker, state.true_goal)


def defender_chase_belief_policy(game, state, belief: Dict[int, float]) -> int:
    most_likely_goal = max(belief, key=belief.get)
    return shortest_path_next_move(game, state.defender, most_likely_goal)


def ambiguity_score_for_move(game, next_node: int, candidate_goals: List[int]) -> float:
    """
    Higher score = more ambiguous.
    We use the negative gap between distances to the candidate goals.
    Smaller gap => more ambiguous => larger score.
    """
    if len(candidate_goals) != 2:
        raise ValueError("This ambiguity score currently assumes exactly 2 candidate goals.")

    g1, g2 = candidate_goals
    d1 = game.shortest_distance(next_node, g1)
    d2 = game.shortest_distance(next_node, g2)

    return -abs(d1 - d2)


def attacker_ambiguous_policy(
    game,
    state,
    ambiguity_weight: float = 2.0,
    progress_weight: float = 1.0,
) -> int:
    """
    Attacker trades off:
    - ambiguity: keep both goals plausible
    - progress: still get closer to the true goal

    score = ambiguity_weight * ambiguity_score - progress_weight * distance_to_true_goal
    """
    neighbors = game.adjacency[state.attacker]
    best_nbr: Optional[int] = None
    best_score = -float("inf")

    for nbr in neighbors:
        ambiguity = ambiguity_score_for_move(game, nbr, game.goals)
        progress = -game.shortest_distance(nbr, state.true_goal)

        score = ambiguity_weight * ambiguity + progress_weight * progress

        if score > best_score:
            best_score = score
            best_nbr = nbr

    if best_nbr is None:
        raise ValueError(f"No legal move from node {state.attacker}")

    return best_nbr