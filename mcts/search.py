'''
The search module contains the MCTS class, which is used to represent
the Monte Carlo Tree Search algorithm.
'''
import numpy as np
from neural_network.anet import ANet
from .node import Node
from .policy import TreePolicy, DefaultPolicy


class MCTS:
    '''
    The MCTS class is used to represent the Monte Carlo Tree Search algorithm.

    Attributes
    ----------
    root_node : Node
        The root node of the search tree.
    n_simulations : int
        The number of simulations.
    '''

    def __init__(
            self,
            root_node: Node,
            n_simulations: int,
            behavior_policy=None,
            nn: ANet = None
    ):
        self.root_node: Node = root_node
        self.n_simulations: int = n_simulations
        self.behavior_policy = behavior_policy
        self.nn = nn

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

    def leaf_evaluation(self, leaf_node: Node) -> int:
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
        if self.behavior_policy is None:
            if self.nn is not None:
                while not leaf_node.is_terminal():
                    if leaf_node.children == []:
                        possible_next_states = leaf_node.state.expand()
                        leaf_node.expand(possible_next_states)
                    node_presentation = leaf_node.state.extract_represenation()
                    distribution = self.mcts.visit_count_distribution()
                    target_dist = self.nn.predict(node_presentation, distribution)
                    i = np.argmax(target_dist)
                    leaf_node = leaf_node.children[i]
                evalution = leaf_node.state.get_value()
            else:
                default_policy = DefaultPolicy()
                evalution: Node = default_policy(leaf_node)
                evalution = evalution.state.get_value()
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

    def visit_count_distribution(self):
        '''
        Returns the visit count distribution of the children of the root node.

        Returns
        -------
        distribution: list
            The visit count distribution of the children of the root node.
        '''
        visit_counts = [child.visits for child in self.root_node.children]
        total_visit_count = sum(visit_counts)
        distribution = [count / total_visit_count for count in visit_counts]
        return distribution

    def __call__(self) -> tuple[Node, list]:
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
        for _ in range(self.n_simulations):
            leaf_node: Node = self.search()
            evaluation = self.leaf_evaluation(leaf_node)
            self.backpropagate(leaf_node, evaluation)
        return self.root_node.get_best_child(), self.visit_count_distribution()
