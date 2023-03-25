from copy import deepcopy
import random
from .node import Node


class MCTS(): 
    def __init__(self, root_node, n_simulations):
        self.root_node: Node = root_node
        self.n_simulations = n_simulations

    def search(self) -> Node:
        curr_root_node: Node = self.root_node
        if curr_root_node.children == []:
            legal_moves = curr_root_node.state.expand()
            curr_root_node.expand(legal_moves)
        curr_root_node = random.choice(curr_root_node.children)
        return curr_root_node
    def leaf_evaluation(self, leaf_node: Node) -> int:
        while not leaf_node.is_terminal():
            if leaf_node.children == []:
                legal_moves = leaf_node.state.expand()
                leaf_node.expand(legal_moves)
            leaf_node = random.choice(leaf_node.children)
        return leaf_node.get_value()
    def backpropagate(self, node: Node, value: int):
        while not node.is_root():
            node.update(value)
            node = node.parent

    def best_move(self) -> Node:
        return max(self.root_node.children, key=lambda node: node.value / node.visits)

    def __call__(self) -> Node:
        for _ in range(self.n_simulations):
            leaf_node = self.search()
            if not leaf_node.is_terminal():
                if leaf_node.children == []:    
                    legal_moves = leaf_node.state.expand()
                    leaf_node.expand(legal_moves)
                node = random.choice(leaf_node.children)
            eval = self.leaf_evaluation(node)
            self.backpropagate(node, eval)
        return self.best_move()
