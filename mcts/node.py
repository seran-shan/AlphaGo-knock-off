'''
This module contains the Node class, which is used to represent a node in the search tree.
'''
from config import BOARD_SIZE
from game import State
import numpy as np
import random


class Node:
    '''
    The Node class is used to represent a node in the search tree.
    '''

    def __init__(self, state: State, parent=None):
        self.state = state
        self.parent: 'Node' = parent
        #intialize children as empty numpy array with the same function as self.children = []:  list['Node'] = []
        self.children: list['Node'] = []
        self.visits: int = 0
        self.value: float = 0
        self.last_child = None

    def add_child(self, child_state) -> 'Node':
        '''
        Add a child node to the current node.

        Parameters
        ----------
        child_state : State
            The state of the child node.

        Returns
        -------
        child_node : Node
            The child node.
        '''
        child_node = Node(child_state, self)
        self.children.append(child_node)
        return child_node

    def add_children(self, child_states):
        '''
        Add multiple child nodes to the current node.

        Parameters
        ----------
        child_states : list of State
            The states of the child nodes.
        '''
        for child_state in child_states:
            self.add_child(child_state)

    def update(self, value: int):
        '''
        Update the value and visits of the current node.

        Parameters
        ----------
        value : int
            The value of the current node.
        '''
        self.visits += 1
        self.value += value

    def is_leaf(self) -> bool:
        '''
        Check if the current node is a leaf node.

        Returns
        -------
        is_leaf : bool
            True if the current node is a leaf node, False otherwise.
        '''
        return len(self.children) == 0

    def is_root(self) -> bool:
        '''
        Check if the current node is the root node.

        Returns
        -------
        is_root : bool
            True if the current node is the root node, False otherwise.
        '''
        return self.parent is None

    def is_terminal(self) -> bool:
        '''
        Check if the current node is a terminal node.

        Returns
        -------
        is_terminal : bool
            True if the current node is a terminal node, False otherwise.
        '''
        return self.state.is_terminal()

    def get_value(self) -> int:
        '''
        Get the value of the current node.

        Returns
        -------
        value : float
            The value of the current node.
        '''
        return self.state.get_value()

    def expand(self, next_states, illegal_state=None):
        '''
        Expand the current node by adding its children.

        Parameters
        ----------
        next_states : list of State
            The states of the child nodes.
        illegal_state : float
            The state of the illegal child node.
        '''
        if illegal_state is not None:
            legal_child_states = [
                state != illegal_state for state in next_states]
            self.add_children(legal_child_states)
        else:
            self.add_children(next_states)

    def visit_count_distribution(self) -> np.ndarray:
        '''
        Returns the visit count distribution of the children of the root node.

        Returns
        -------
        distribution: list
            The visit count distribution of the children of the root node.
        '''
        visit_counts = [0] * BOARD_SIZE**2

        for child in self.children:
            prev_action = child.state.get_previous_action()
            index = prev_action[0] * BOARD_SIZE + prev_action[1]
            visit_counts[index] = child.visits
        total_visit_count = sum(visit_counts)
        distribution = np.array(
            [count / total_visit_count for count in visit_counts])
        return distribution

    def get_best_child(self) -> 'Node':
        '''
        Get the best child node from the current node.

        Returns

        -------
        best_child : Node
            The best child node. If player is 1, then the best child is a maximum, otherwise it is a minimum.
        '''
        # return max value of random of one of the largest values
        total_visits = sum(child.visits for child in self.children)
        max_ratio = max(child.visits / total_visits for child in self.children)
        best_children = [child for child in self.children if child.visits / total_visits == max_ratio]
        return random.choice(best_children)


        # return max(self.children, key=lambda node: node.visits / sum(child.visits for child in self.children))

    def __str__(self) -> str:
        return f'Node({self.state}, {self.visits}, {self.value})'

    