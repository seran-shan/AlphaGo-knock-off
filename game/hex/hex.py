import numpy as np
from disjoint_set import DisjointSet
import copy


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

    def get_last_move(self):
        """
        Return the last move made in the current state

        """
        pass

    def get_value(self):
        """
        Return the value of the current state

        """
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
        if move not in self.get_legal_moves():
            return False
        return True

    def get_winner(self):
        return self.winner

    def get_value(self):
        return self.get_winner()

    def is_terminal(self):
        return self.get_winner() is not None

    def set_winner(self, winner):
        self.winner = int(winner)

    def change_player(self):
        self.player = 1 - self.player

    def get_last_move(self):
        return self.last_move

    def set_last_move(self, move):
        self.last_move = move

    def get_legal_moves(self):
        """
        Return a list of legal moves. A move is a tuple (x, y) where x and y are the coordinates of the move.
        """
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves

    def make_move(self, move):
        """
        Make a move on the board, change the player to move, and check if the game is over.
        """
        x, y = move
        self.board[x][y] = -1 if self.player == 0 else 1
        if self.player == 0:
            self.disjoint_set_player_0.find(move)
            for neighbour in self.get_adjecent_neighbours(x, y):
                if self.board[neighbour[0]][neighbour[1]] == -1:
                    self.disjoint_set_player_0.union(move, neighbour)
        else:
            self.disjoint_set_player_1.find(move)
            for neighbour in self.get_adjecent_neighbours(x, y):
                if self.board[neighbour[0]][neighbour[1]] == 1:
                    self.disjoint_set_player_1.union(move, neighbour)
        self.set_last_move(move)
        self.check_winner()
        self.change_player()

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


if __name__ == "__main__":
    game = Hex(3)
    while not game.is_terminal():
        print(game.board)
        move = game.get_move()
        game.make_move(move)
    print("Winner: ", game.get_winner())
