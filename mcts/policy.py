'''
The policy module contains the TreePolicy and DefaultPolicy classes.
'''
import random
import numpy as np
from .node import Node


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
        node: Node
            The current node.
        c_punt: float
            The exploration constant.

        Returns
        -------
        max_child_node: Node
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
        node: Node
            The current node.
        c_punt: float
            The exploration constant.

        Returns
        -------
        min_child_node: Node
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
        child_node: Node
            The child node.

        Returns
        -------
        value: float
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
        node: Node
            The current node.

        Returns
        -------
        next_node: Node
            The next node.
        '''
        if self.turn % 2 == 0:
            next_node = self.maximize()
        else:
            next_node = self.minimize()
        return next_node


class DefaultPolicy:
    '''
    The DefaultPolicy class is used to represent the default policy of the
    Monte Carlo Tree Search algorithm. This should be the policy that is used
    to evaluate the leaf nodes. As the default policy, the target policy is
    used, since we are using on-policy Monte Carlo Tree Search.
    '''

    def __init__(self, node: Node):
        self.node: Node = node

    def __call__(self, target_value) -> int:
        '''
        Using the target policy to evaluate the leaf node. Randomly selecting child
        nodes until the game is finished.

        Parameters
        ----------
        node: Node
            The leaf node.
        target_value: int
            The value of the target state.
        '''
        curr_node = self.node
        while not curr_node.value == target_value:
            curr_node = random.choice(curr_node.children)
        return curr_node.value
