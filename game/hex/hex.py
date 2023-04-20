'''
This module exports the Hex game class. 
'''

import copy
import random
import numpy as np
from disjoint_set import DisjointSet


class State:
    '''
    The State interface is used to represent a state in the search tree.
    '''

    def is_terminal(self) -> bool:
        '''
        Check if the current state is a terminal state.

        Returns
        -------
        is_terminal : bool
            True if the current state is a terminal state, False otherwise.
        '''
        pass

    def expand(self):
        """
        Expand the current node.

        Return nodes from the intial state by performing all legal moves without modifing the intial state

        """
        pass

    def get_value(self):
        """
        Return the value of the current state

        """
        pass

    def get_previous_action(self):
        """
        Return the previous state

        """
        pass

    def get_legal_actions(self):
        """
        Return the legal actions from the current state

        """
        pass

    def produce_successor_state(self, action):
        """
        Return the successor state after performing the action

        """
        pass

    def extract_representation(self):
        """
        Return the representation of the state

        """
        pass

    def extract_flatten_state(self):
        """
        Return the state

        """
        pass

    def expand_random(self):
        '''
        Expand the current node by performing a random move.
        '''
        pass

    def expand_index(self, index):
        '''
        Expand the current node by performing the move at the given index.
        '''
        pass


class Hex:
    """
    Hex game class. 

    Parameters
    ----------
    size: int
        The size of the board.

    board: list of list of int
        The board.
        Hex board is represented as a to dimensional list. 
        The values of the list are 0, 1, or -1.
        0 means the position is empty.
        1 means the position is occupied by the maximizer.
        -1 means the position is occupied by the minimizer.

    player : int
        The player to move next.

    winner : int or None
        The winner of the game. None if the game is not over. 1 for the maximizer, -1 for the minimizer.

    """

    def __init__(self, size):
        self.size = size
        self.board = np.array([[0 for i in range(size)] for j in range(size)])
        self.player = 0
        self.winner = None
        self.last_move = None
        # Intialize legal moves
        self.legal_moves = {(i, j) for i in range(size) for j in range(size)}
        # Intialize neighbors of each position
        self.neighbors = {(i, j): self.get_adjecent_neighbours(i, j)
                          for i in range(size) for j in range(size)}
        # Disjoint set to check if there is a path from one side to the other
        self.disjoint_set_player_0 = DisjointSet()
        self.disjoint_set_player_1 = DisjointSet()

    def get_move(self):
        """
        Get a move from the player. 

        Returns 
        -------
        move : tuple of int 
            A move consists of an x and y coordinate on the board.
        """
        x = int(input("Enter x coordinate: "))
        y = int(input("Enter y coordinate: "))
        move = (x, y)
        if not self.validate_move((x, y)):
            print("Invalid move")
            return self.get_move()
        return move

    def validate_move(self, move):
        """
        Check if a move is valid.

        Parameters
        ----------
        move : tuple of int
            The move to check.

        """
        return move in self.legal_moves

    def get_winner(self):
        '''
        Return the winner of the game.

        Returns
        -------
        winner : int or None
            The winner of the game. None if the game is not over. 1 for the maximizer, -1 for the minimizer.
        '''
        return self.winner

    def get_value(self):
        '''
        Return the value of the current state.

        Returns
        -------
        value : int
            The value of the current state.
        '''
        return self.get_winner()

    def is_terminal(self):
        '''
        Check if the current state is a terminal state.

        Returns
        -------
        is_terminal : bool
            True if the current state is a terminal state, False otherwise.
        '''
        return self.get_winner() is not None

    def set_winner(self, winner):
        '''
        Set the winner of the game.

        Parameters
        ----------
        winner : int or None
            The winner of the game. None if the game is not over. 1 for the maximizer, -1 for the minimizer.
        '''
        self.winner = int(winner)

    def change_player(self):
        '''
        Change the player to move.
        '''
        self.player = 1 - self.player

    def get_last_move(self):
        '''
        Return the last move made in the current state.

        Returns
        -------
        last_move : tuple of int
            The last move made in the current state.
        '''
        return self.last_move

    def set_last_move(self, move):
        '''
        Set the last move made in the current state.

        Parameters
        ----------
        move : tuple of int
            The last move made in the current state.
        '''
        self.last_move = move

    def get_previous_action(self):
        '''
        Return the previous action.
        '''
        return self.get_last_move()

    def get_legal_moves(self):
        """
        Return a list of legal moves. A move is a tuple (x, y) where x and y are the coordinates of the move.
        """
        return self.legal_moves

    def get_legal_actions(self):
        """
        Return a list of legal actions. An action is a tuple (x, y) where x and y are the coordinates of the action.
        """
        return self.get_legal_moves()

    def make_move(self, move):
        """
        Make a move on the board, change the player to move, and check if the game is over.
        """
        x, y = move
        self.board[x][y] = -1 if self.player == 0 else 1
        self.legal_moves.remove(move)
        if self.player == 0:
            self.disjoint_set_player_0.find(move)
            for neighbour in self.neighbors[x, y]:
                if self.board[neighbour[0]][neighbour[1]] == -1:
                    self.disjoint_set_player_0.union(move, neighbour)
        else:
            self.disjoint_set_player_1.find(move)
            for neighbour in self.neighbors[x, y]:
                if self.board[neighbour[0]][neighbour[1]] == 1:
                    self.disjoint_set_player_1.union(move, neighbour)
        self.set_last_move(move)
        self.check_winner()
        self.change_player()

    def produce_successor_state(self, action):
        '''
        Produce a successor state by making a move.
        '''
        self.make_move(action)

    def expand(self):
        """
        Return a list of all possible next states.
        """
        states = []
        for move in self.get_legal_moves():
            state = copy.deepcopy(self)
            state.make_move(move)
            states.append(state)
        return states

    def expand_random(self):
        """
        Return a random successor state.
        """
        move = random.choice(list(self.get_legal_moves()))
        state = copy.deepcopy(self)
        state.make_move(move)
        return state

    def expand_index(self, index):
        '''
        Return the successor state at index.
        '''
        move = list(self.get_legal_moves())[index]
        state = copy.deepcopy(self)
        state.make_move(move)
        return state

    def check_winner(self):
        """
        Check if the game is over, and set the winner if the game is over.

        Returns
        -------
        winner : int or None
            The winner of the game. None if the game is not over.
        """
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][0] == -1 and self.board[j][self.size - 1] == -1:
                    if self.disjoint_set_player_0.connected((i, 0), (j, self.size - 1)):
                        self.set_winner(-1)

                if self.board[0][i] == 1 and self.board[self.size - 1][j] == 1:
                    if self.disjoint_set_player_1.connected((0, i), (self.size - 1, j)):
                        self.set_winner(1)

    def get_adjecent_neighbours(self, x, y):
        """
        Return a list of adjecent neighbours of a position on the board.

        Parameters
        ----------
        x : int
            The x coordinate of the position.
        y : int
            The y coordinate of the position.

        Returns
        -------
        neighbours : list of tuple of int
            A list of adjecent neighbours.
        """

        neighbours = []
        if x > 0:
            neighbours.append((x - 1, y))
        if x > 0 and y < self.size - 1:
            neighbours.append((x-1, y+1))
        if x < self.size - 1:
            neighbours.append((x + 1, y))
        if y > 0:
            neighbours.append((x, y - 1))
        if y > 0 and x < self.size - 1:
            neighbours.append((x+1, y-1))
        if y < self.size - 1:
            neighbours.append((x, y + 1))
        return neighbours

    def extract_representation(self, training=True):
        '''
        Extract a representation of the current state, to feed it to a neural network.
        '''
        flat_board = self.board.flatten()
        player_to_move = np.array([self.player if self.player == 1 else -1])

        # if self.last_move is not None:
        #     last_move = np.array(
        #         [self.last_move[0] * self.size + self.last_move[1]])
        # else:
        #     last_move = np.array([-1])

        board_representation = np.hstack(
            [flat_board, player_to_move])
        if training:
            return board_representation
        return np.expand_dims(board_representation, axis=0)

    def extract_flatten_state(self):
        return self.board.flatten()

    def draw(self):
        '''
        Draw the current state.
        '''
        even_row = True
        l = self.size * 2 - 1
        for i in range(l):
            if even_row:
                for j in range(i):
                    # white spaces used for aligning the rows
                    print(" ", end="")
                # print the row with nodes
                even = True
                for j in range(l):
                    if even:
                        if self.board[i // 2][j // 2] == -1:
                            print("X", end="")
                        elif self.board[i // 2][j // 2] == 1:
                            print("+", end="")
                        else:
                            print(self.board[i // 2][j // 2], end="")
                    else:
                        print(" - ", end="")
                    even = not even
            else:
                # print the seperation row
                # print the blank space to align the graph
                for j in range(i):
                    print(" ", end="")
                even = True
                for j in range(l):
                    if even:
                        print("\\ ", end="")
                    else:
                        print("/ ", end="")
                    even = not even
            even_row = not even_row
            print()
