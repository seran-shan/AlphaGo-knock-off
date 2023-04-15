'''
The policy module contains the TreePolicy and DefaultPolicy classes.
'''
import random
import numpy as np

from neural_network.anet import ANet
from .node import Node


class TreePolicy:
    '''
    The TreePolicy class is used to represent the tree policy of the
    Monte Carlo Tree Search algorithm.
    '''

    def __init__(self, node: Node, c_punt: float = np.sqrt(2)):
        self.node: Node = node
        self.c_punt: float = c_punt

    def maximize(self) -> Node:
        '''
        Select the child node with the highest value.

        Returns
        -------
        max_child_node: Node
            The child node with the highest value.
        '''
        return max(self.node.children, key=self.calculate_value)

    def minimize(self) -> Node:
        '''
        Select the child node with the lowest value.

        Returns
        -------
        min_child_node: Node
            The child node with the lowest value.
        '''
        return min(self.node.children, key=self.calculate_value)

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

        epsilon = 1

        if child_node.visits == 0:
            q_value = 0
        else:
            q_value = child_node.value / child_node.visits
        exploration_bonus = self.c_punt * \
            np.sqrt(np.log(self.node.visits + epsilon) /
                    (child_node.visits + epsilon))
        return q_value - exploration_bonus if child_node.state.player == 1 else q_value + exploration_bonus

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
        if self.node.state.player == 1:
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

    def __call__(self, curr_node: Node) -> Node:
        '''
        Using the target policy to evaluate the leaf node. Randomly selecting child
        nodes until the game is finished.

        Parameters
        ----------
        node: Node
            The leaf node.
        '''
        while not curr_node.is_terminal():
            if curr_node.children == []:
                possible_next_states = curr_node.state.expand()
                curr_node.expand(possible_next_states)
            curr_node = random.choice(curr_node.children)
        return curr_node


class TargetPolicy:
    '''
    The TargetPolicy class is used to represent the target policy of the
    Monte Carlo Tree Search algorithm. This should be the policy that is used
    to evaluate the leaf nodes. As the target policy, the target policy is
    used, since we are using on-policy Monte Carlo Tree Search.
    '''

    def __init__(self, neural_network: ANet):
        self.neural_network = neural_network

    def __call__(self, leaf_node: Node, distribution: callable) -> Node:
        '''
        Using the target policy to evaluate the leaf node. Randomly selecting child
        nodes until the game is finished.

        Parameters
        ----------
        node: Node
            The leaf node.
        '''
        while not leaf_node.is_terminal():
            if leaf_node.children == []:
                possible_next_states = leaf_node.state.expand()
                leaf_node.expand(possible_next_states)
            state_representation = leaf_node.state.extract_representation()
            target_dist = self.neural_network.predict(
                state_representation, distribution())
            i = np.argmax(target_dist)
            leaf_node = leaf_node.children[i]
        return leaf_node
