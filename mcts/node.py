'''
This module contains the Node class, which is used to represent a node in the search tree.
'''


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
