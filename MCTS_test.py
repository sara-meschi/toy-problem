import math
import random
import networkx as nx
import numpy as np

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state  # Current positions and beliefs
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0

    def uct_select(self, exploration=1.41):
        """Standard Upper Confidence Bound for Trees."""
        return max(self.children, key=lambda c: (c.wins / c.visits) + 
                   exploration * math.sqrt(math.log(self.visits) / c.visits))

    def is_fully_expanded(self, possible_moves):
        return len(self.children) == len(possible_moves)
    
    def mcts_search(game, player_type, iterations=500):

    # """
    # player_type: 'attacker' or 'defender'
    # # iterations: Number of MCTS iterations to perform 
    # """
    
    # Sample a goal for the simulation if we are the Defender
        sim_goal = game.true_goal
        if player_type == 'defender':
            # Defender samples a candidate goal based on current belief distribution
            sim_goal = game.goals[np.random.choice(len(game.goals), p=game.beliefs)]

        root = MCTSNode(state={'a_pos': game.attacker_pos, 'd_pos': game.defender_pos})

        for _ in range(iterations):
            node = root
            # 1. SELECTION
            # (Simplified: logic to traverse down to a leaf)
            
            # 2. EXPANSION & SIMULATION (Playout)
            # Randomly walk until termination [cite: 15, 16, 17]
            payoff = rollout(game, node.state, sim_goal)
            
            # 3. BACKPROPAGATION
            # Update node.visits and node.wins based on the payoff [cite: 18, 19]
            backpropagate(node, payoff, player_type)

        return root.uct_select(exploration=0).move # Best move (no exploration)
    
    def rollout(game, state, goal):
        curr_a = state['a_pos']
        curr_d = state['d_pos']
        
        for _ in range(15): # Limit depth to avoid infinite loops
            # Attacker moves 1 hop towards goal [cite: 11]
            a_neighbors = list(game.graph.neighbors(curr_a))
            curr_a = random.choice(a_neighbors)
            
            # Defender moves 1 hop towards attacker [cite: 11]
            d_neighbors = list(game.graph.neighbors(curr_d))
            curr_d = random.choice(d_neighbors)
            
            # Check termination [cite: 15, 16, 17]
            if curr_a == goal:
                return nx.shortest_path_length(game.graph, curr_d, goal) # [cite: 19]
            if nx.shortest_path_length(game.graph, curr_d, curr_a) <= 1:
                return -nx.shortest_path_length(game.graph, curr_a, goal) # [cite: 19]
                
        return 0 # Draw/Timeout
    # change