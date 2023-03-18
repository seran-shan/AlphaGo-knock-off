'''
This module contains the Nim game class.
'''


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


class Nim(State):
    '''
    Nim game class.

    parameters
    ----------
    pieces : list of int
       number of pieces in the pile

    maxPieces : int
        The maximum number of pieces that can be removed in a move.

    player : int
        The player who is to move next.

    winner : int or None
        The winner of the game. None if the game is not over. 1 for the maximizer, -1 for the minimizer.

    '''

    def __init__(self, N: int, K: int, player: int = 0):
        self.pieces = N
        self.max_pieces = K
        self.player = player
        self.winner = None

    def get_move(self):
        '''
        Get a move from the player.

        Returns
        -------
        move : tuple of int
            The move to make.
        '''
        move = int(input("How many pieces do you want to remove?"))
        if not self.validate_move(move):
            print("Invalid move")
            return self.get_move()
        return move

    def validate_move(self, move):
        '''
        Check if a move is valid.

        Parameters
        ----------
        move : tuple of int
            The move to check.

        Returns
        -------
        valid : bool
            True if the move is valid, False otherwise.
        '''
        if (move > self.max_pieces or move > self.pieces) or move <= 0:
            return False
        return True

    def get_winner(self):
        '''
        Return the winner of the game.
        '''
        return self.winner

    def is_terminal(self):
        '''
        Check if the game is over.
        '''
        return self.get_winner() is not None

    def set_winner(self, winner):
        '''
        Set the winner of the game.
        '''
        self.winner = winner

    def change_player(self):
        '''
        Change the player to move.
        '''
        self.player = 1 - self.player

    def get_legal_moves(self):
        '''
        Return a list of legal moves.

        A move is a tuple (pile, number) where pile is the pile index
        and number is the number of objects to remove from the pile.
        '''
        moves = []
        for i in range(1, self.max_pieces + 1):
            if self.validate_move(i):
                moves.append(i)
        return moves

    def make_move(self, move):
        '''
        Make a move. Check winner after performing move.

        Parameters
        ----------
        move : int
            The move to make.
        '''
        number = move
        self.pieces -= number
        self.change_player()
        self.check_winner()

        return Nim(self.pieces, self.max_pieces, self.player)

    def expand(self):
        """
        Expand the current node.

        Return nodes from the intial state by performing all legal moves without modifing the intial state

        """

        child_states = []
        for move in self.get_legal_moves():
            child_state = Nim(self.pieces, self.max_pieces, self.player)
            child_state.make_move(move)
            child_states.append(child_state)
        return child_states

    def check_winner(self):
        '''
        Check if the game is over and set the winner.
        '''
        if (self.pieces == 0):
            if self.player == 0:
                self.set_winner(1)
            else:
                self.set_winner(-1)

    def __str__(self):
        '''
        Return a string representation of the game.
        '''
        return f'NimGame:\nPieces: {self.pieces}  \nMax Pieces to take: {self.max_pieces} \nPlayer {self.player} to move'
