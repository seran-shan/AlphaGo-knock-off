'''
The search module contains the MCTS class, which is used to represent 
the Monte Carlo Tree Search algorithm.
'''
import numpy as np
from .node import Node


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
        Perform the Monte Carlo Tree Search algorithm.
        '''
        turn = 0 if player == 1 else 1
        curr_root_node = self.root_node

        while len(curr_root_node.children) > 0:
            curr_root_node = self.tree_policy(c_punt, turn, curr_root_node)
            turn += 1
        return curr_root_node


class TreePolicy:
    '''
    The TreePolicy class is used to represent the tree policy of the 
    Monte Carlo Tree Search algorithm.
    '''

    def __init__(self, node: Node, c_punt: float, turn: int):
        self.node: Node = node
        self.c_punt: float = c_punt
        self.turn: int = turn

    def maximize(self) -> Node:
        '''
        Select the child node with the highest value.

        Parameters
        ----------
        node : Node
            The current node.
        c_punt : float
            The exploration constant.

        Returns
        -------
        max_child_node : Node
            The child node with the highest value.
        '''
        max_value = -np.inf
        max_child_node: Node = None
        for child_node in self.node.children:
            value = self.calculate_value(child_node)
            if value >= max_value:
                max_value = value
                max_child_node = child_node
        return max_child_node

    def minimize(self) -> Node:
        '''
        Select the child node with the lowest value.

        Parameters
        ----------
        node : Node
            The current node.
        c_punt : float
            The exploration constant.

        Returns
        -------
        min_child_node : Node
            The child node with the lowest value.
        '''
        min_value = np.inf
        min_child_node: Node = None
        for child_node in self.node.children:
            value = self.calculate_value(child_node)
            if value <= min_value:
                min_value = value
                min_child_node = child_node
        return min_child_node

    def calculate_value(self, child_node: Node) -> float:
        '''
        Calculate the value of the child node.

        Parameters
        ----------
        child_node : Node
            The child node.
        c_punt : float
            The exploration constant.

        Returns
        -------
        value : float
            The value of the child node.
        '''
        q_value = child_node.value / child_node.visits
        n_value = np.log(child_node.parent.visits) / child_node.visits
        exploration_bonus = self.c_punt * np.sqrt(n_value)

        return q_value + exploration_bonus

    def __call__(self) -> Node:
        '''
        Using the tree policy to select the next node.
        The tree policy is based on the min-max tree policy.

        Parameters
        ----------
        node : Node
            The current node.

        Returns
        -------
        next_node : Node
            The next node.
        '''
        if self.turn % 2 == 0:
            next_node = self.maximize()
        else:
            next_node = self.minimize()
        return next_node
