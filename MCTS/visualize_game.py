# import matplotlib.pyplot as plt
# import numpy as np
# import networkx as nx
# from MCTS.toy_sim_MCTS import ReachAvoidGame, mcts_decision


# def run_and_visualize():
#     # 1. Initialize Game
#     size = 5
#     game = ReachAvoidGame(size=size)
    
#     a_path = [game.attacker_pos]
#     d_path = [game.defender_pos]
#     belief_history = [game.beliefs.copy()]
    
#     print(f"True Goal: {game.true_goal}")

#     # 2. Run Simulation
#     steps = 20
#     final_payoff = 0
#     for s in range(steps):
#         a_move = mcts_decision(game, 'attacker', iterations=150)
#         d_move = mcts_decision(game, 'defender', iterations=150)
        
#         old_a = game.attacker_pos
#         game.attacker_pos, game.defender_pos = a_move, d_move
#         game.update_beliefs(old_a, a_move)
        
#         a_path.append(game.attacker_pos)
#         d_path.append(game.defender_pos)
#         belief_history.append(game.beliefs.copy())

        
#         payoff = game.get_payoff(game.attacker_pos, game.defender_pos, game.true_goal)

#         print(f"Step {s}: A at {game.attacker_pos}, D at {game.defender_pos}, Beliefs: {game.beliefs}")
#         if payoff is not None:
#             final_payoff = payoff
#             print(f"Game Over! Final Payoff: {final_payoff}")
#             break

#     # 3. Create Plots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

#     # --- Plot 1: Trajectories ---
#     # Draw Grid
#     for x in range(size):
#         ax1.axhline(x, lw=1, color='gray', alpha=0.3)
#         ax1.axvline(x, lw=1, color='gray', alpha=0.3)

#     # Convert paths to numpy for easy plotting
#     ap = np.array(a_path)
#     dp = np.array(d_path)

#     # Plot Goals
#     for i, g in enumerate(game.goals):
#         color = 'gold' if g == game.true_goal else 'gray'
#         label = f"Goal {i} (TRUE)" if g == game.true_goal else f"Goal {i} (Candidate)"
#         ax1.scatter(g[1], g[0], s=300, marker='*', c=color, edgecolors='black', label=label, zorder=5)

#     # Plot Paths
#     ax1.plot(ap[:, 1], ap[:, 0], 'r-o', label='Attacker Path', markersize=8, alpha=0.8)
#     ax1.plot(dp[:, 1], dp[:, 0], 'b-s', label='Defender Path', markersize=8, alpha=0.8)
    
#     # Starting positions
#     ax1.text(a_path[0][1], a_path[0][0], " Start A", color='red', fontweight='bold')
#     ax1.text(d_path[0][1], d_path[0][0], " Start D", color='blue', fontweight='bold')

#     ax1.set_title(f"Game Trajectory (Payoff: {final_payoff})")
#     ax1.set_xlim(-0.5, size-0.5)
#     ax1.set_ylim(-0.5, size-0.5)
#     ax1.invert_yaxis() # Match matrix/grid indexing
#     ax1.legend()

#     # --- Plot 2: Beliefs over Time ---
#     beliefs = np.array(belief_history)
#     ax2.plot(beliefs[:, 0], label="Belief in Goal 0 (0,0)", color='green', lw=2)
#     ax2.plot(beliefs[:, 1], label="Belief in Goal 1 (4,4)", color='orange', lw=2)
#     ax2.set_xlabel("Time Step")
#     ax2.set_ylabel("Probability")
#     ax2.set_title("Defender's Belief Evolution")
#     ax2.set_ylim(-0.05, 1.05)
#     ax2.grid(True, linestyle='--')
#     ax2.legend()

#     plt.tight_layout()
#     plt.savefig("simulation_results.png") # Saves for your presentation
#     plt.show()

# if __name__ == "__main__":
#     run_and_visualize()

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from toy_sim_MCTS_v2 import ReachAvoidGame, mcts_search

def run_and_visualize():
    # 1. Initialize Game
    size = 10
    game = ReachAvoidGame(size=size)
    
    a_path = [game.a_pos]
    d_path = [game.d_pos]
    belief_history = [game.beliefs.copy()]
    
    print(f"True Goal: {game.true_goal}")

    # 2. Run Simulation
    steps = 50
    final_payoff = 0
    for s in range(steps):

        # PHASE 1: ATTACKER'S TURN
        
        game.turn = 'attacker'
        a_move = mcts_search(game, root_turn='attacker', iterations=150)
        
        old_a = game.a_pos
        game.a_pos = a_move
        
        # Defender observes the move and updates belief IMMEDIATELY
        likelihoods = game.calculate_likelihoods(old_a, game.a_pos)
        raw_post = game.beliefs * likelihoods
        game.beliefs = raw_post / np.sum(raw_post)
        
        # Check if Attacker won by stepping on the goal before Defender moves
        payoff = game.get_payoff(game.a_pos, game.d_pos, game.true_goal)
        if payoff is not None:
            a_path.append(game.a_pos)
            d_path.append(game.d_pos)
            belief_history.append(game.beliefs.copy())
            
            final_payoff = payoff
           
            if payoff > 0:
                print(f"Step {s} (Attacker Reached Goal!): A at {game.a_pos}, D at {game.d_pos}")
            else:
                print(f"Step {s} (Attacker Blundered and got Tagged!): A at {game.a_pos}, D at {game.d_pos}")
            # ---------------
            print(f"Game Over! Final Payoff: {final_payoff}")
            break

        # PHASE 2: DEFENDER'S TURN

        game.turn = 'defender'
        # Defender searches using the newly updated belief
        d_move = mcts_search(game, root_turn='defender', iterations=150)
        game.d_pos = d_move
        
        # Record keeping for visualization
        a_path.append(game.a_pos)
        d_path.append(game.d_pos)
        belief_history.append(game.beliefs.copy())

        print(f"Step {s}: A at {game.a_pos}, D at {game.d_pos}, Beliefs: {game.beliefs}")
        
       # Check if Defender won by tagging the Attacker
        payoff = game.get_payoff(game.a_pos, game.d_pos, game.true_goal)
        if payoff is not None:
            final_payoff = payoff
            print(f"Step {s} (Defender Tagged Attacker!): A at {game.a_pos}, D at {game.d_pos}")
            print(f"Game Over! Final Payoff: {final_payoff}")
            break

    # 3. Create Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Trajectories ---
    # Draw Grid
    for x in range(size):
        ax1.axhline(x, lw=1, color='gray', alpha=0.3)
        ax1.axvline(x, lw=1, color='gray', alpha=0.3)

    # Convert paths to numpy for easy plotting
    ap = np.array(a_path)
    dp = np.array(d_path)

    # Plot Goals
    for i, g in enumerate(game.goals):
        color = 'gold' if g == game.true_goal else 'gray'
        label = f"Goal {i} (TRUE)" if g == game.true_goal else f"Goal {i} (Candidate)"
        ax1.scatter(g[1], g[0], s=300, marker='*', c=color, edgecolors='black', label=label, zorder=5)

    # Plot Paths
    ax1.plot(ap[:, 1], ap[:, 0], 'r-o', label='Attacker Path', markersize=8, alpha=0.8)
    ax1.plot(dp[:, 1], dp[:, 0], 'b-s', label='Defender Path', markersize=8, alpha=0.8)
    
    # Starting positions
    ax1.text(a_path[0][1], a_path[0][0], " Start A", color='red', fontweight='bold')
    ax1.text(d_path[0][1], d_path[0][0], " Start D", color='blue', fontweight='bold')

    ax1.set_title(f"Game Trajectory (Payoff: {final_payoff})")
    ax1.set_xlim(-0.5, size-0.5)
    ax1.set_ylim(-0.5, size-0.5)
    ax1.invert_yaxis() # Match matrix/grid indexing
    ax1.legend()

    # --- Plot 2: Beliefs over Time ---
    beliefs = np.array(belief_history)
    ax2.plot(beliefs[:, 0], label="Belief in Goal 0 (0,0)", color='green', lw=2)
    ax2.plot(beliefs[:, 1], label="Belief in Goal 1 (4,4)", color='orange', lw=2)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Probability")
    ax2.set_title("Defender's Belief Evolution")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle='--')
    ax2.legend()

    plt.tight_layout()
    plt.savefig("simulation_results.png") 
    plt.show()

if __name__ == "__main__":
    run_and_visualize()