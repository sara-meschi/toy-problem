from graph_game import GraphGame
from beliefs import attacker_move_posterior


def main():
    adjacency = {
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4, 6],
        4: [1, 3, 5, 7],
        5: [2, 4, 8],
        6: [3, 7],
        7: [4, 6, 8],
        8: [5, 7],
    }

    game = GraphGame(
        adjacency=adjacency,
        attacker_start=0,
        defender_start=8,
        goals=[2, 6],
        max_steps=10,
        tag_radius=1,
    )

    prior = {2: 0.5, 6: 0.5}

    # Attacker observed to move 0 -> 1
    posterior_1 = attacker_move_posterior(
        game=game,
        prior=prior,
        current_attacker_node=0,
        observed_next_attacker_node=1,
        candidate_goals=[2, 6],
        beta=2.0,
    )
    print("Posterior after observing 0 -> 1:", posterior_1)

    # Attacker observed to move 0 -> 3
    posterior_2 = attacker_move_posterior(
        game=game,
        prior=prior,
        current_attacker_node=0,
        observed_next_attacker_node=3,
        candidate_goals=[2, 6],
        beta=2.0,
    )
    print("Posterior after observing 0 -> 3:", posterior_2)


if __name__ == "__main__":
    main()