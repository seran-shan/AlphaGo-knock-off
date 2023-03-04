'''
The search module contains the MCTS class, which is used to represent
the Monte Carlo Tree Search algorithm.
'''
from .node import Node
from .policy import TreePolicy


class MCTS:
    '''
    The MCTS class is used to represent the Monte Carlo Tree Search algorithm.

    Attributes
    ----------
    root_node : Node
        The root node of the search tree.
    n_simulations : int
        The number of simulations.
    default_policy : callable
        The default policy.
    tree_policy : callable
        The tree policy.
    rollout_policy : callable
        The rollout policy.
    '''

    def __init__(
            self,
            root_node: Node,
            n_simulations: int,
            default_policy: callable,
            tree_policy: 'TreePolicy',
            rollout_policy=None
    ):
        self.root_node: Node = root_node
        self.n_simulations: int = n_simulations
        self.default_policy: callable = default_policy
        self.tree_policy: TreePolicy = tree_policy
        self.rollout_policy: callable = rollout_policy

    def search(self, player, c_punt) -> Node:
        '''
        Performing tree search with the tree policy.
        '''
        turn = 0 if player == 1 else 1
        curr_root_node = self.root_node

        while len(curr_root_node.children) > 0:
            curr_root_node = self.tree_policy(c_punt, turn, curr_root_node)
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
        return self.default_policy(leaf_node.state)

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
