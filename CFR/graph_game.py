from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque


Node = int


@dataclass(frozen=True)
class GameState:
    attacker: Node
    defender: Node
    true_goal: Node
    time_step: int = 0
    terminal: bool = False
    outcome: Optional[str] = None  # "attacker_goal", "defender_tag", or None
    tag_node: Optional[Node] = None


class GraphGame:
    def __init__(
        self,
        adjacency: Dict[Node, List[Node]],
        attacker_start: Node,
        defender_start: Node,
        goals: List[Node],
        max_steps: int = 20,
        tag_radius: int = 1,
    ) -> None:
        if len(goals) < 2:
            raise ValueError("Need at least two candidate goals.")

        self.adjacency = adjacency
        self.attacker_start = attacker_start
        self.defender_start = defender_start
        self.goals = goals
        self.max_steps = max_steps
        self.tag_radius = tag_radius

        self._validate_graph()
        self.distances = self._all_pairs_shortest_paths()

    def _validate_graph(self) -> None:
        for node, neighbors in self.adjacency.items():
            for nbr in neighbors:
                if nbr not in self.adjacency:
                    raise ValueError(f"Neighbor {nbr} of node {node} is not in graph.")

    def _bfs_distances_from(self, start: Node) -> Dict[Node, int]:
        dist = {start: 0}
        q = deque([start])

        while q:
            current = q.popleft()
            for nbr in self.adjacency[current]:
                if nbr not in dist:
                    dist[nbr] = dist[current] + 1
                    q.append(nbr)
        return dist

    def _all_pairs_shortest_paths(self) -> Dict[Node, Dict[Node, int]]:
        return {node: self._bfs_distances_from(node) for node in self.adjacency}

    def shortest_distance(self, a: Node, b: Node) -> int:
        return self.distances[a][b]

    def initial_state(self, true_goal: Node) -> GameState:
        if true_goal not in self.goals:
            raise ValueError("true_goal must be one of the candidate goals.")

        state = GameState(
            attacker=self.attacker_start,
            defender=self.defender_start,
            true_goal=true_goal,
            time_step=0,
            terminal=False,
            outcome=None,
            tag_node=None,
        )

        # In case they start already within tag radius
        if self._is_tagged(state.attacker, state.defender):
            return GameState(
                attacker=state.attacker,
                defender=state.defender,
                true_goal=state.true_goal,
                time_step=0,
                terminal=True,
                outcome="defender_tag",
                tag_node=state.attacker,
            )

        return state

    def get_legal_actions(self, player: str, state: GameState) -> List[Node]:
        if state.terminal:
            return []

        if player == "attacker":
            return list(self.adjacency[state.attacker])
        if player == "defender":
            return list(self.adjacency[state.defender])

        raise ValueError("player must be 'attacker' or 'defender'")

    def _is_tagged(self, attacker_node: Node, defender_node: Node) -> bool:
        return self.shortest_distance(attacker_node, defender_node) <= self.tag_radius

    def is_terminal(self, state: GameState) -> bool:
        return state.terminal

    def transition(
        self,
        state: GameState,
        attacker_action: Node,
        defender_action: Node,
    ) -> GameState:
        if state.terminal:
            return state

        if attacker_action not in self.adjacency[state.attacker]:
            raise ValueError(f"Illegal attacker action: {attacker_action}")
        if defender_action not in self.adjacency[state.defender]:
            raise ValueError(f"Illegal defender action: {defender_action}")

        next_attacker = attacker_action
        next_defender = defender_action
        next_time = state.time_step + 1

        # Terminal rule 1: defender tags attacker
        # if self._is_tagged(next_attacker, next_defender):
        #     return GameState(
        #         attacker=next_attacker,
        #         defender=next_defender,
        #         true_goal=state.true_goal,
        #         time_step=next_time,
        #         terminal=True,
        #         outcome="defender_tag",
        #         tag_node=next_attacker,
        #     )

        # # Terminal rule 2: attacker reaches true goal
        # if next_attacker == state.true_goal:
        #     return GameState(
        #         attacker=next_attacker,
        #         defender=next_defender,
        #         true_goal=state.true_goal,
        #         time_step=next_time,
        #         terminal=True,
        #         outcome="attacker_goal",
        #         tag_node=None,
        #     )

                # Terminal rule 1: attacker reaches true goal
        if next_attacker == state.true_goal:
            return GameState(
                attacker=next_attacker,
                defender=next_defender,
                true_goal=state.true_goal,
                time_step=next_time,
                terminal=True,
                outcome="attacker_goal",
                tag_node=None,
            )

        # Terminal rule 2: defender tags attacker
        if self._is_tagged(next_attacker, next_defender):
            return GameState(
                attacker=next_attacker,
                defender=next_defender,
                true_goal=state.true_goal,
                time_step=next_time,
                terminal=True,
                outcome="defender_tag",
                tag_node=next_attacker,
            )

        # Safety horizon
        if next_time >= self.max_steps:
            return GameState(
                attacker=next_attacker,
                defender=next_defender,
                true_goal=state.true_goal,
                time_step=next_time,
                terminal=True,
                outcome="timeout",
                tag_node=None,
            )

        return GameState(
            attacker=next_attacker,
            defender=next_defender,
            true_goal=state.true_goal,
            time_step=next_time,
            terminal=False,
            outcome=None,
            tag_node=None,
        )

    def terminal_payoff(self, state: GameState) -> float:
        """
        Attacker is maximizer, defender is minimizer.

        Problem statement:
        - If attacker reaches goal:
            payoff = distance(defender, true_goal)
        - If defender tags:
            payoff = - distance(tagging_point, true_goal)

        Here tagging_point is approximated as attacker's node when tag occurs.
        """
        if not state.terminal:
            raise ValueError("Payoff only defined for terminal states.")

        if state.outcome == "attacker_goal":
            return float(self.shortest_distance(state.defender, state.true_goal))

        if state.outcome == "defender_tag":
            if state.tag_node is None:
                raise ValueError("Tag outcome must include tag_node.")
            return -float(self.shortest_distance(state.tag_node, state.true_goal))

        # Timeout fallback
        return -100.0