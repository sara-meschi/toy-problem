import networkx as nx
import numpy as np

class ReachAvoidGame:
    def __init__(self, grid_size=(5,5)):
        self.graph = nx.grid_2d_graph(*grid_size) # 
        self.goals = [(0,0), (4,4)] # Candidate goals 
        self.true_goal = self.goals[np.random.choice(len(self.goals))] # 
        self.attacker_pos = (2,0)
        self.defender_pos = (2,4)
        self.beliefs = np.array([0.5, 0.5]) # Prior probability 

    # 
    
    def update_beliefs(self, old_pos, new_pos):
        beta = 1.0  # Rationality parameter: higher = more optimal attacker
        likelihoods = []

        for goal in self.goals:
            # How much did this move reduce the distance to THIS goal?
            old_dist = nx.shortest_path_length(self.graph, old_pos, goal)
            new_dist = nx.shortest_path_length(self.graph, new_pos, goal)
            
            # Likelihood: e^(beta * distance_reduction)
            # If distance decreased, likelihood is higher.
            likelihood = np.exp(beta * (old_dist - new_dist))
            likelihoods.append(likelihood)

        # Apply Bayes: Posterior = (Prior * Likelihood) / Normalizer
        raw_posterior = self.beliefs * np.array(likelihoods)
        self.beliefs = raw_posterior / np.sum(raw_posterior)

    # def check_termination(self):
    #     # Check if distance <= 1 (Tag) or Attacker at True Goal [cite: 15, 16, 17]
    #     pass

    def check_termination(self):
        # 1. Check if Attacker reached the true goal [cite: 16]
        if self.attacker_pos == self.true_goal:
            # Payoff: Defender's distance to goal 
            payoff = nx.shortest_path_length(self.graph, self.defender_pos, self.true_goal)
            return True, payoff
        
        # 2. Check if Defender tags the Attacker (distance <= 1 hop) [cite: 15, 17]
        dist_to_attacker = nx.shortest_path_length(self.graph, self.defender_pos, self.attacker_pos)
        if dist_to_attacker <= 1:
            # Payoff: Negative distance from tagging point to goal 
            payoff = -nx.shortest_path_length(self.graph, self.attacker_pos, self.true_goal)
            return True, payoff
            
        return False, None
    
    def step(self, attacker_move, defender_move):
        # Record old position for belief update
        old_attacker_pos = self.attacker_pos
        
        # Execute moves [cite: 11]
        self.attacker_pos = attacker_move
        self.defender_pos = defender_move
        
        # Update Defender's belief based on observing history [cite: 14, 22]
        self.update_beliefs(old_attacker_pos, self.attacker_pos)
        
        # Check if someone won [cite: 17]
        done, payoff = self.check_termination()
        return done, payoff