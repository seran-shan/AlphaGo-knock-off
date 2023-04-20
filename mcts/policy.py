'''
The policy module contains the TreePolicy and DefaultPolicy classes.
'''
import random
import numpy as np

from neural_network.anet import ANet
from .node import Node
from time import time
import copy


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
        return q_value - exploration_bonus if child_node.state.player == 0 else q_value + exploration_bonus

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
            next_state = curr_node.state.expand_random()
            curr_node.add_child(next_state)
            for child in curr_node.children:
                if child.state == next_state:
                    curr_node = child
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

    def __call__(self, leaf_node: Node, epsilon: float) -> Node:
        '''
        Using the target policy to evaluate the leaf node. Randomly selecting child
        nodes until the game is finished.

        Parameters
        ----------
        node: Node
            The leaf node.
        '''
        while not leaf_node.is_terminal():
            state_representation = leaf_node.state.extract_representation(False)
            target_dist = self.neural_network.predict(state_representation)
            flatten_state = leaf_node.state.extract_flatten_state()
            legal_action = [1 if flatten_state[i] ==
                            0 else 0 for i in range(len(flatten_state))]

            target_dist = np.array(target_dist) * np.array(legal_action)
            target_dist = target_dist[target_dist != 0]
            i = np.argmax(target_dist)
            if random.uniform(0,1) < epsilon:

                next_state = leaf_node.state.expand_random()
                leaf_node.add_child(next_state)
                for child in leaf_node.children:
                    if child.state == next_state:
                        leaf_node = child
            else:
                next_state = leaf_node.state.expand_index(i)
                leaf_node.add_child(next_state)
                for child in leaf_node.children:
                    if child.state == next_state:
                        leaf_node = child
        return leaf_node
