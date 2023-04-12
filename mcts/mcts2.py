from copy import deepcopy
import random
from .node import Node
import numpy as np


class MCTS(): 
    def __init__(self, root_node, n_simulations):
        '''
        The MCTS class is used to represent the Monte Carlo Tree Search algorithm.

        Attributes
        ----------
        root_node : Node
            The root node of the search tree.  
        n_simulations : int
            The number of simulations.
        '''
        self.root_node: Node = root_node
        self.n_simulations = n_simulations

    def search(self) -> Node:
        '''
        Performing tree search form the root node.

        Returns
        -------
        curr_root_node: Node
            The root node of the search tree.
        '''

        curr_root_node: Node = self.root_node
        if curr_root_node.children == []:
            legal_moves = curr_root_node.state.expand()
            curr_root_node.expand(legal_moves)
        curr_root_node = random.choice(curr_root_node.children)
        return curr_root_node
    

    def leaf_evaluation(self, leaf_node: Node) -> int:
        '''
        Estimating the value of a leaf node in the tree by doing a rollout simulation

        Parameters 
        ----------
        leaf_node: Node
            The leaf node.

        Returns
        -------
        value: int
            The value of the leaf node.
        '''


        while not leaf_node.is_terminal():
            if leaf_node.children == []:
                legal_moves = leaf_node.state.expand()
                leaf_node.expand(legal_moves)
            leaf_node = random.choice(leaf_node.children)
        return leaf_node.get_value()
    

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
        node.update(value)
        if node.parent is not None:
            self.backpropagate(node.parent, value)
    
    def calculate_value(self, child_node: Node, c_punt: float) -> float:
        '''
        Calculate the value of a particular node. 

        Attributes
        ----------
        node: Node
            The current node.
        c_punt: float
            The exploration constant.
        player: int

        Returns
        -------
        value: float
            The value of the node.
        '''
        
        if child_node.visits == 0: 
            q = 0
        else:
            q = child_node.value / child_node.visits
        exploration_bonus = c_punt * np.sqrt(np.log(self.root_node.visits) / child_node.visits)
        return q - exploration_bonus if child_node.state.player == 1 else q + exploration_bonus
    

    
    def best_move(self) -> Node:
        '''
        Get the best move from the root node. If player is 1, then the best move is a maximum, otherwise it is a minimum.

        Returns

        -------
        best_move : Node
            The best move from the root node.
        '''

        if self.root_node.state.player == 1:
            return max(self.root_node.children, key=lambda node: self.calculate_value(node, np.sqrt(2)))
        else:
           return min(self.root_node.children, key=lambda node: self.calculate_value(node, np.sqrt(2))) 


        # if self.root_node.state.player == 1:
        #     return max(self.root_node.children, key=lambda node: node.value / (node.visits + 1))
        # else:
        #    return min(self.root_node.children, key=lambda node: node.value / (node.visits + 1)) 
        
    def __call__(self) -> Node:
        '''
        Performing a Monte Carlo Tree Search.

        Returns
        -------
        best_move: Node
            The best move from the root node.
        '''
        
        for _ in range(self.n_simulations):
            leaf_node = self.search()
            if not leaf_node.is_terminal():
                if leaf_node.children == []:    
                    legal_moves = leaf_node.state.expand()
                    leaf_node.expand(legal_moves)
                node = random.choice(leaf_node.children)
            eval = self.leaf_evaluation(node)
            self.backpropagate(node, eval)
        return self.best_move()
