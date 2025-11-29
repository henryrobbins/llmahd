from __future__ import annotations
import math
from typing import Any


class MCTSNode:

    def __init__(
        self,
        algorithm: str,
        code: str,
        obj: float,
        depth: int = 0,
        parent: MCTSNode | None = None,
        visit: int = 0,
        raw_info: Any | None = None,
        Q: float = 0,
        is_root: bool = False,
    ) -> None:
        self.algorithm = algorithm
        self.code = code
        self.parent = parent
        self.depth = depth
        self.children: list[MCTSNode] = []
        self.children_info: list[Any] = []
        self.visits = visit
        self.subtree: list[MCTSNode] = []
        self.raw_info = raw_info
        self.Q = Q
        self.reward = -1 * obj
        self.is_root = is_root

    def add_child(self, child_node: MCTSNode) -> None:
        self.children.append(child_node)

    def __repr__(self) -> str:
        return f"MCTSNode(algorithm={self.algorithm}, Q={self.Q:.2f}, visits={self.visits})"


class MCTS:
    def __init__(self, root_answer: str) -> None:
        self.exploration_constant_0 = 0.1
        self.alpha = 0.5
        self.max_depth = 10
        self.epsilon = 1e-10
        self.discount_factor = 1
        self.q_min = 0
        self.q_max = -10000
        self.rank_list = []

        self.root = MCTSNode(
            algorithm=root_answer, code=root_answer, depth=0, obj=0, is_root=True
        )

        # Logs
        self.critiques = []
        self.refinements = []
        self.rewards = []
        self.selected_nodes = []

    def backpropagate(self, node: MCTSNode) -> None:
        if node.Q not in self.rank_list:
            self.rank_list.append(node.Q)
            self.rank_list.sort()
        self.q_min = min(self.q_min, node.Q)
        self.q_max = max(self.q_max, node.Q)
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = (
                parent.Q * (1 - self.discount_factor)
                + best_child_Q * self.discount_factor
            )
            parent.visits += 1
            if parent.code != "Root" and parent.parent.code == "Root":
                parent.subtree.append(node)
            parent = parent.parent

    def uct(self, node: MCTSNode, eval_remain: float) -> float:
        self.exploration_constant = (self.exploration_constant_0) * eval_remain
        return (node.Q - self.q_min) / (
            self.q_max - self.q_min
        ) + self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1) / node.visits
        )

    def is_fully_expanded(self, node: MCTSNode) -> bool:
        return (
            len(node.children) >= self.max_children
            or any(child.Q > node.Q for child in node.children)
            or node.code == "Root"
        )
