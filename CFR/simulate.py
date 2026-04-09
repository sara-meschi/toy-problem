from __future__ import annotations

from graph_game import GraphGame
from beliefs import attacker_move_posterior
from policies import attacker_shortest_path_policy, defender_chase_belief_policy


def simulate_one_game(game: GraphGame, true_goal: int, beta: float = 2.0):
    state = game.initial_state(true_goal=true_goal)
    belief = {g: 1.0 / len(game.goals) for g in game.goals}

    trajectory = [
        {
            "time": state.time_step,
            "attacker": state.attacker,
            "defender": state.defender,
            "belief": belief.copy(),
            "outcome": state.outcome,
        }
    ]

    while not game.is_terminal(state):
        current_attacker_node = state.attacker

        # Attacker moves first
        attacker_action = attacker_shortest_path_policy(game, state)

        # Defender observes attacker move and updates belief
        belief = attacker_move_posterior(
            game=game,
            prior=belief,
            current_attacker_node=current_attacker_node,
            observed_next_attacker_node=attacker_action,
            candidate_goals=game.goals,
            beta=beta,
        )

        # Defender moves based on updated belief
        defender_action = defender_chase_belief_policy(game, state, belief)

        # Apply transition
        state = game.transition(state, attacker_action, defender_action)

        trajectory.append(
            {
                "time": state.time_step,
                "attacker": state.attacker,
                "defender": state.defender,
                "belief": belief.copy(),
                "outcome": state.outcome,
            }
        )

    payoff = game.terminal_payoff(state)
    return state, payoff, trajectory


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

    for true_goal in [2, 6]:
        final_state, payoff, trajectory = simulate_one_game(game, true_goal=true_goal, beta=2.0)

        print("\n" + "=" * 60)
        print(f"True goal: {true_goal}")
        print("Trajectory:")
        for step in trajectory:
            print(step)

        print("Final state:", final_state)
        print("Payoff:", payoff)


if __name__ == "__main__":
    main()