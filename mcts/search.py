'''
The search module contains the MCTS class, which is used to represent
the Monte Carlo Tree Search algorithm.
'''
import time
import numpy as np
from neural_network.anet import ANet
from .node import Node
from .policy import TargetPolicy, TreePolicy, DefaultPolicy
import random

class MCTS:
    '''
    The MCTS class is used to represent the Monte Carlo Tree Search algorithm.

    Attributes
    ----------
    root_node : Node
        The root node of the search tree.
    n_simulations : int
        The number of simulations.
    neural_network : ANet
        The neural network.
    '''

    def __init__(
            self,
            root_node: Node,
            n_simulations: int,
            time_limit: int,
            neural_network: ANet = None,

    ):
        self.root_node: Node = root_node
        self.n_simulations: int = n_simulations
        self.time_limit: int = time_limit
        self.neural_network = neural_network


    def search(self) -> Node:
        '''
        Performing tree search with the tree policy.

        Returns
        -------
        curr_root_node: Node
            The root node of the search tree.
        '''

        curr_root_node: Node = self.root_node

        if curr_root_node.children == []:
            legal_moves = curr_root_node.state.expand()
            curr_root_node.expand(legal_moves)

        tree_policy = TreePolicy(self.root_node)
        curr_root_node = tree_policy()

        return curr_root_node

    def leaf_evaluation(self, leaf_node: Node, epsilon: float) -> int:
        '''
        Estimating the value of a leaf node in the tree by doing a rollout simulation 
        using the default policy from the leaf node’s state to a final state.

        Parameters
        ----------
        leaf_node: Node
            The leaf node.

        Returns
        -------
        evalution: int
            The value of the leaf node.
        '''
        if self.neural_network:
            target_policy = TargetPolicy(self.neural_network)
            evalution = target_policy(
                leaf_node, epsilon).state.get_value()
        else:
            default_policy = DefaultPolicy()
            evalution = default_policy(leaf_node).state.get_value()
        return evalution

    def backpropagate(self, node: Node, value: int):
        '''
        Backpropagate the evaluation of a final state back up the tree, updating relevant
        data at all nodes and edges on the path from the final state to the tree root.

        Parameters
        ----------
        node: Node
            The current node.
        value: int
            The value of the current node.
        '''
        node.update(value)
        if node.parent is not None:
            self.backpropagate(node.parent, value)

    def __call__(self, epsilon: float = None) -> tuple[Node, list]:
        '''
        Performing a Monte Carlo Tree Search using the tree policy to select the next node.

        Parameters
        ----------
        node: Node
            The current node.

        Returns
        -------
        best_child: Node
            The best child of the root node.
        distribution: list
            The visit count distribution of the children of the root node.
        '''
        start_time = time.time()
        simulations = 0

        while (time.time() - start_time < self.time_limit or simulations < self.n_simulations) and (time.time() - start_time < 8):
            test_time = time.time()
            leaf_node: Node = self.search()
            evaluation = self.leaf_evaluation(leaf_node, epsilon)
            self.backpropagate(leaf_node, evaluation)
            simulations += 1
        print("Simulations: ", simulations)
        return self.root_node.get_best_child(), self.root_node.visit_count_distribution()