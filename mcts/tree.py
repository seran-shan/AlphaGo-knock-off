'''
This module contains the Tree class, which is used to represent the search tree
'''
import numpy as np
from node import Node


class Tree:
    '''
    The Tree class is used to represent the search tree.
    '''

    def __init__(self, state):
        self.root = Node(state)

    def get_root(self) -> Node:
        '''
        Get the root node of the tree.

        Returns
        -------
        root : Node
            The root node of the tree.
        '''
        return self.root

    def set_root(self, root: Node):
        '''
        Set the root node of the tree.

        Parameters
        ----------
        root : Node
            The root node of the tree.
        '''
        self.root = root

    def tree_search(self, next_player: int, exploration_constant: float, board) -> Node:
        '''
        Perform a tree search from the root node.

        Parameters
        ----------
        root_node : Node
            The root node of the tree.
        next_player : int
            The player whose turn it is.
        exploration_constant : float
            The exploration constant used in the tree policy.
        board : HexBoard
        '''
        turn = 0 if next_player == 1 else 1
        iter_root_node = self.root

        # Traverse the tree until a leaf node is reached
        while len(iter_root_node.children) > 0:
            # Maximize player
            if turn % 2 == 0:
                max_value = -np.inf
                max_child = None
                # Iterating over children on a given node, ending each for loop with the child with the max value edge to the current node
                for child in iter_root_node.children:
                    tree_policy = iter_root_node.calculate_tree_policy(
                        exploration_constant, child)
                    if tree_policy >= max_value:
                        max_value = tree_policy
                        max_child = child
                iter_root_node = max_child
            # Minimize player
            else:
                min_value = np.inf
                min_child = None
               # Iterating over children on a given node, ending each for loop with the child with the min value edge to the current node
                for child in iter_root_node.children:
                    tree_policy = iter_root_node.calculate_tree_policy(
                        -exploration_constant, child)
                    if tree_policy <= min_value:
                        min_value = tree_policy
                        min_child = child
                iter_root_node = min_child

            # Update the board state to reflect the current node's move
            if iter_root_node is not None:
                board.update_state(iter_root_node.state)

            turn += 1

        return iter_root_node
