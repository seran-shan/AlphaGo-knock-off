

class Nim: 
    """
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

    """

    def __init__(self, N, K): 
        self.pieces = N
        self.maxPieces = K
        self.player = 0
        self.winner = None
    
    def get_move(self):
        """
        Get a move from the player.

        Returns
        -------
        move : tuple of int
            The move to make.
        """
        move = int(input("How many pieces do you want to remove?"))
        if not self.validate_move(move):
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

        Returns
        -------
        valid : bool
            True if the move is valid, False otherwise.
        """
        if (move > self.maxPieces or move > self.pieces) or move <= 0:
            return False
        return True
    def get_winner(self): 
        return self.winner
    
    def set_winner(self, winner): 
        self.winner = winner

    def change_player(self):
        self.player = 1 - self.player

    def get_legal_moves(self): 
        """
        Return a list of legal moves.

        A move is a tuple (pile, number) where pile is the pile index
        and number is the number of objects to remove from the pile.
        """
        moves = []
        for pile in range(len(self.piles)):
            for number in range(1, self.piles[pile] + 1):
                moves.append((pile, number))
        return moves
    
    def make_move(self, move):
        """
        Make a move. Check winner after performing move.

        Parameters
        ----------
        move : int
            The move to make.
        """
        number = move
        self.pieces -= number
        self.change_player()
        self.check_winner()
    
    def check_winner(self):
        """
        Check if the game is over and set the winner.
        """
        if (self.pieces == 0):
            if self.player == 0:
                self.set_winner(1)
            else:
                self.set_winner(-1)

    def __str__(self):
        """
        Return a string representation of the game.
        """
        return "NimGame:\nPieces: {}  \nMax Pieces to take: {} \nPlayer {} to move".format(self.pieces, self.maxPieces, self.player)


    def main(self):
        """
        Play the game.
        """
        while self.get_winner() is None:
            print(str(self))

            move = self.get_move()
            self.make_move(move)
        print("Winner:", self.get_winner())

if __name__ == "__main__":
    game = Nim(5,2)
    game.main()

