'''
This module contains the Node class, which is used to represent a node in the search tree.
'''
from game.hex.hex import State


class Node:
    '''
    The Node class is used to represent a node in the search tree.
    '''

    def __init__(self, state: State, parent=None):
        self.state = state
        self.parent: 'Node' = parent
        self.children: list['Node'] = []
        self.visits: int = 0
        self.value: float = 0
        self.last_move = None

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
        '''
        return self.state.get_value()
    
    def get_last_move(self) -> 'str':
        '''
        Get the last move made in the current node.

        Returns
        -------
        last_move : str
            The last move made in the current node.

        '''
        return self.state.get_last_move()

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

    def get_best_child(self) -> 'Node':
        '''
        Get the best child node of the current node.

        Returns
        -------
        best_child : Node
            The best child node.
        '''
        best_child = max(self.children, key=lambda child: child.value)
        return best_child

    def __str__(self) -> str:
        return f'Node({self.state}, {self.visits}, {self.value})'
