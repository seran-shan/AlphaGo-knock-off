'''
This module contains the Node class, which is used to represent a node in the search tree.
'''
import numpy as np

class Node:
    '''
    The Node class is used to represent a node in the search tree.
    '''
    def __init__(self, state, parent=None):
        self.state = state
        self.parent: 'Node' = parent
        self.children: list['Node'] = []
        self.visits: int = 0
        self.value: float = 0

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

    def calculate_tree_policy(self, exploration_constant: float, child: 'Node') -> float:
        '''
        Calculate the tree policy of the current node.

        Parameters
        ----------
        exploration_constant : float
            The exploration constant used in the tree policy.
        child : Node
            The child node.

        Returns
        -------
        tree_policy : float
            The tree policy of the current node.
        '''
        return child.value / child.visits + exploration_constant * np.sqrt(np.log(self.visits) / child.visits)
