# import networkx as nx
# import numpy as np
# import math
# import random

# # --- GAME ENVIRONMENT ---
# class ReachAvoidGame:
#     def __init__(self, size=5):
#         self.graph = nx.grid_2d_graph(size, size)
#         self.goals = [(0, 0), (size-1, size-1)]  # Candidate goals [cite: 12]
#         self.true_goal = random.choice(self.goals)  # Hidden from Defender [cite: 12, 14]
#         self.attacker_pos = (size-1, 0)
#         self.defender_pos = (0, size-1)
#         self.beliefs = np.array([0.5, 0.5])  # Prior probability [cite: 13]
#         self.history = {'attacker': [self.attacker_pos], 'defender': [self.defender_pos]}

#     def get_payoff(self, a_pos, d_pos, goal):
#         """Calculates payoff based on game rules [cite: 18, 19, 20]"""
#         if a_pos == goal:
#             # Attacker reaches goal: Payoff is Defender's distance to goal
#             return nx.shortest_path_length(self.graph, d_pos, goal)
        
#         dist_a_d = nx.shortest_path_length(self.graph, d_pos, a_pos)
#         if dist_a_d <= 1:
#             # Defender tags Attacker: Payoff is negative distance from tag point to goal
#             return -nx.shortest_path_length(self.graph, a_pos, goal)
#         return None

#     def update_beliefs(self, old_a, new_a):
#         """Updates Defender's guess using Bayes' Rule [cite: 21, 22]"""
#         beta = 1.5  # Rationality parameter
#         likelihoods = []
#         for g in self.goals:
#             # P(move | Goal) - is the attacker moving closer to this goal?
#             d_old = nx.shortest_path_length(self.graph, old_a, g)
#             d_new = nx.shortest_path_length(self.graph, new_a, g)
#             likelihoods.append(np.exp(beta * (d_old - d_new)))
        
#         raw_post = self.beliefs * np.array(likelihoods)
#         self.beliefs = raw_post / np.sum(raw_post)

# # --- MCTS ENGINE ---
# class MCTSNode:
#     def __init__(self, a_pos, d_pos, move=None, parent=None):
#         self.a_pos = a_pos
#         self.d_pos = d_pos
#         self.move = move
#         self.parent = parent
#         self.children = []
#         self.wins = 0
#         self.visits = 0

#     def uct_select(self):
#         return max(self.children, key=lambda c: (c.wins / c.visits) + 1.41 * math.sqrt(math.log(self.visits) / c.visits))

# def mcts_decision(game, player_type, iterations=150):
#     # Defender samples a goal based on current beliefs [cite: 14, 27]
#     sim_goal = game.true_goal if player_type == 'attacker' else \
#                game.goals[np.random.choice(len(game.goals), p=game.beliefs)]
    
#     root = MCTSNode(game.attacker_pos, game.defender_pos)
    
#     for _ in range(iterations):
#         node = root
#         # 1. Selection & Expansion
#         while True:
#             payoff = game.get_payoff(node.a_pos, node.d_pos, sim_goal)
#             if payoff is not None: break # Terminal
            
#             # Simple expansion: pick a neighbor
#             moves = list(game.graph.neighbors(node.a_pos if player_type == 'attacker' else node.d_pos))
#             if len(node.children) < len(moves):
#                 m = moves[len(node.children)]
#                 new_a = m if player_type == 'attacker' else node.a_pos
#                 new_d = node.d_pos if player_type == 'attacker' else m
#                 child = MCTSNode(new_a, new_d, move=m, parent=node)
#                 node.children.append(child)
#                 node = child
#                 break
#             node = node.uct_select()

#         # 2. Simulation (Rollout)
#         curr_a, curr_d = node.a_pos, node.d_pos
#         for _ in range(10): # Depth limit
#             p = game.get_payoff(curr_a, curr_d, sim_goal)
#             if p is not None: 
#                 payoff = p
#                 break
#             curr_a = random.choice(list(game.graph.neighbors(curr_a)))
#             curr_d = random.choice(list(game.graph.neighbors(curr_d)))
#             payoff = 0 # Draw
            
#         # 3. Backpropagation [cite: 20]
#         while node:
#             node.visits += 1
#             node.wins += payoff if player_type == 'attacker' else -payoff
#             node = node.parent

#     return max(root.children, key=lambda c: c.visits).move

# # --- MAIN EXECUTION ---
# if __name__ == "__main__":
#     game = ReachAvoidGame(size=5)
#     print(f"True Goal: {game.true_goal}")

#     for step in range(20):
#         # Decisions 
#         a_move = mcts_decision(game, 'attacker')
#         d_move = mcts_decision(game, 'defender')

#         # Update World [cite: 11, 22]
#         old_a = game.attacker_pos
#         game.attacker_pos, game.defender_pos = a_move, d_move
#         game.update_beliefs(old_a, a_move)
        
#         print(f"Step {step}: A at {game.attacker_pos}, D at {game.defender_pos}, Beliefs: {game.beliefs}")

#         # Check Win Conditions [cite: 15, 16, 17]
#         result = game.get_payoff(game.attacker_pos, game.defender_pos, game.true_goal)
#         if result is not None:
#             print(f"Game Over! Final Payoff: {result}")
#             break

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from MCTS.toy_sim_MCTS_v2 import ReachAvoidGame, mcts_search

def run_and_visualize():
    # 1. Initialize Game
    size = 5
    game = ReachAvoidGame(size=size)
    
    a_path = [game.a_pos]
    d_path = [game.d_pos]
    belief_history = [game.beliefs.copy()]
    
    print(f"True Goal: {game.true_goal}")

    # 2. Run Simulation
    steps = 20
    final_payoff = 0
    for s in range(steps):
        # ==========================================
        # PHASE 1: ATTACKER'S TURN
        # ==========================================
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
            print(f"Step {s} (Attacker Won): A at {game.a_pos}, D at {game.d_pos}, Beliefs: {game.beliefs}")
            print(f"Game Over! Final Payoff: {final_payoff}")
            break

        # ==========================================
        # PHASE 2: DEFENDER'S TURN
        # ==========================================
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
    plt.savefig("simulation_results.png") # Saves for your presentation
    plt.show()

if __name__ == "__main__":
    run_and_visualize()