# toy-problem


```markdown
# Reach-Avoid Game with Incomplete Information

This repository studies a graph-based reach-avoid game with incomplete information using two approaches:

- **MCTS**
- **CFR**

An **Attacker** moves toward a hidden true goal, while a **Defender** tries to infer that goal and intercept the attacker. The Defender does not know the true goal, but knows the candidate goals and prior probabilities.

## Approaches

### MCTS
The MCTS solver uses a belief-state representation with:
- attacker position
- defender position
- defender belief over goals
- turn information

It updates belief after attacker motion and uses heuristic rollouts to simulate future play.

### CFR
The CFR solver treats the problem as an extensive-form imperfect-information game. The attacker’s hidden goal defines its private type, and the defender acts based on partial information using regret minimization over repeated iterations.

## Motivation

This work is inspired by deceptive path planning and Bayesian game formulations, especially:
- *Deception by Motion: The Eater and the Mover Game* (2023)
- *Deceptive Path Planning: A Bayesian Game Approach* (2025)

The game in this repo is a modified version where the Defender is an embodied moving agent that can tag the Attacker.

## Notes

These implementations are intended as research prototypes rather than exact reproductions of the reference papers’ equilibrium solutions.