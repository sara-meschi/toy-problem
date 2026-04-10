from graph_game import GraphGame


def main():
    # Simple undirected graph
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

    state = game.initial_state(true_goal=2)
    print("Initial:", state)

    print("Attacker legal:", game.get_legal_actions("attacker", state))
    print("Defender legal:", game.get_legal_actions("defender", state))

    # Example move
    state = game.transition(state, attacker_action=1, defender_action=7)
    print("After step 1:", state)

    state = game.transition(state, attacker_action=2, defender_action=4)
    print("After step 2:", state)

    if game.is_terminal(state):
        print("Terminal payoff:", game.terminal_payoff(state))


if __name__ == "__main__":
    main()