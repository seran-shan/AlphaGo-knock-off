'''
The search module contains the MCTS class, which is used to represent
the Monte Carlo Tree Search algorithm.
'''
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
    ):
        self.root_node: Node = root_node
        self.n_simulations: int = n_simulations

    def search(self, player, c_punt) -> Node:
        '''
        Performing tree search with the tree policy.
        '''
        turn = 0 if player == 1 else 1
        curr_root_node = self.root_node

        while len(curr_root_node.children) > 0:
            curr_root_node = TreePolicy(c_punt, turn, curr_root_node)
            turn += 1
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
        value: int
            The value of the leaf node.
        '''
        return DefaultPolicy(leaf_node.state)(leaf_node.value)

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
        while node is not None:
            node.update(value)
            node = node.parent

    def __call__(self, player, c_punt) -> Node:
        '''
        Performing a Monte Carlo Tree Search using the tree policy to select the next node.
        The tree policy is based on the min-max tree policy.

        Parameters
        ----------
        node: Node
            The current node.

        Returns
        -------
        next_node: Node
            The next node.
        '''
        for _ in range(self.n_simulations):
            leaf_node = self.search(player, c_punt)
            value = self.leaf_evaluation(leaf_node)
            self.backpropagate(leaf_node, value)
        return self.root_node.get_best_child()
