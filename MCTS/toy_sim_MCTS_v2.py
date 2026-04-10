import networkx as nx
import numpy as np
import math
import random

class ReachAvoidGame:
    def __init__(self, size=10):
        self.graph = nx.grid_2d_graph(size, size)
        self.goals = [(0, 0), (size-1, size-1)]
        self.true_goal = random.choice(self.goals)
        
        # State is now clearly defined
        self.a_pos = (size-1, 0)
        # self.d_pos = (0, size-1)
        self.d_pos = (size//2, size//2) # Be careful about odd number grid size!!!
        self.beliefs = np.array([0.5, 0.5])
        self.turn = 'attacker' # Alternating turns

    def get_payoff(self, a_pos, d_pos, goal):
        if a_pos == goal:
            return nx.shortest_path_length(self.graph, d_pos, goal)
        if nx.shortest_path_length(self.graph, d_pos, a_pos) <= 1:
            return -nx.shortest_path_length(self.graph, a_pos, goal)
        return None

    def calculate_likelihoods(self, old_a, new_a):
        """
        Feedback Implementation: Ambiguous vs Explicit motion.
        Instead of raw distance, we classify the move's intent.
        """
        likelihoods = []
        for g in self.goals:
            d_old = nx.shortest_path_length(self.graph, old_a, g)
            d_new = nx.shortest_path_length(self.graph, new_a, g)
            
            if d_new < d_old:
                prob = 0.8  # Move is consistent with this goal
            elif d_new == d_old:
                prob = 0.5  # Move is ambiguous regarding this goal
            else:
                prob = 0.1  # Move is explicitly away from this goal
            likelihoods.append(prob)
        return np.array(likelihoods)

    def get_next_state(self, a_pos, d_pos, beliefs, turn, action):
        """Simulates a state transition, INCLUDING belief updates."""
        if turn == 'attacker':
            new_a = action
            new_d = d_pos
            # Defender observes attacker move and updates belief
            likelihoods = self.calculate_likelihoods(a_pos, new_a)
            raw_post = beliefs * likelihoods
            new_beliefs = raw_post / np.sum(raw_post)
            next_turn = 'defender'
        else:
            new_a = a_pos
            new_d = action
            new_beliefs = beliefs # Defender moving doesn't change their own belief of the attacker
            next_turn = 'attacker'
            
        return new_a, new_d, new_beliefs, next_turn

# --- BELIEF-STATE MCTS ---
class MCTSNode:
    def __init__(self, a_pos, d_pos, beliefs, turn, move=None, parent=None):
        self.a_pos = a_pos
        self.d_pos = d_pos
        self.beliefs = beliefs  # BELIEF IS NOW IN THE STATE
        self.turn = turn        # SEQUENTIAL TURNS
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    def uct_select(self):
        # Attacker maximizes, Defender minimizes
        exploration = 1.41
        if self.turn == 'attacker':
            return max(self.children, key=lambda c: (c.wins / c.visits) + exploration * math.sqrt(math.log(self.visits) / c.visits))
        else:
            return min(self.children, key=lambda c: (c.wins / c.visits) - exploration * math.sqrt(math.log(self.visits) / c.visits))

def heuristic_rollout(game, a_pos, d_pos, sim_goal, depth=30):
    """Feedback Implementation: Better than random rollout."""
    curr_a, curr_d = a_pos, d_pos
    for _ in range(depth):
        p = game.get_payoff(curr_a, curr_d, sim_goal)
        if p is not None: return p
        
        # Attacker heuristics: 80% go to goal, 20% random (mimics mixing shortest/deceptive paths)
        a_moves = list(game.graph.neighbors(curr_a))
        if random.random() < 0.8:
            curr_a = min(a_moves, key=lambda m: nx.shortest_path_length(game.graph, m, sim_goal))
        else:
            curr_a = random.choice(a_moves)
            
        # Defender heuristics: 80% chase attacker, 20% random
        d_moves = list(game.graph.neighbors(curr_d))
        if random.random() < 0.8:
            curr_d = min(d_moves, key=lambda m: nx.shortest_path_length(game.graph, m, curr_a))
        else:
            curr_d = random.choice(d_moves)
            
    return 0 # Timeout/Draw

def mcts_search(game, root_turn, iterations=400):
    sim_goal = game.true_goal if root_turn == 'attacker' else \
               game.goals[np.random.choice(len(game.goals), p=game.beliefs)]
               
    root = MCTSNode(game.a_pos, game.d_pos, game.beliefs, root_turn)
    
    for _ in range(iterations):
        node = root
        
        # 1. Selection & Expansion
        while True:
            payoff = game.get_payoff(node.a_pos, node.d_pos, sim_goal)
            if payoff is not None: break
            
            # Determine whose moves to expand
            pos = node.a_pos if node.turn == 'attacker' else node.d_pos
            moves = list(game.graph.neighbors(pos))
            
            if len(node.children) < len(moves):
                m = moves[len(node.children)]
                # Transition to next state using the updated logic
                new_a, new_d, new_b, next_turn = game.get_next_state(node.a_pos, node.d_pos, node.beliefs, node.turn, m)
                child = MCTSNode(new_a, new_d, new_b, next_turn, move=m, parent=node)
                node.children.append(child)
                node = child
                break
            node = node.uct_select()

        # 2. Simulation (Using structured rollouts)
        if payoff is None:
            payoff = heuristic_rollout(game, node.a_pos, node.d_pos, sim_goal)
            
        # 3. Backpropagation
        while node:
            node.visits += 1
            node.wins += payoff # Attacker wants positive, Defender wants negative
            node = node.parent

    # Return best move (Max for attacker, Min for defender)
    if root_turn == 'attacker':
        return max(root.children, key=lambda c: c.wins/c.visits).move
    else:
        return min(root.children, key=lambda c: c.wins/c.visits).move