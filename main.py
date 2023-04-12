'''
Main file for the Nim game.
'''
from game import Nim, Hex
from mcts import MCTS, Node
from mcts.mcts2 import MCTS as MCTS2



def main():
    '''
    Play Hex with a human and AI.
    '''
    game = Hex(5)
    root_node = Node(game)

    while not game.is_terminal():
        print(game.board)  # Display the game state
        if game.player == 0:
            move = game.get_move()
            print("Human move: ", move)
        else:
            mcts = MCTS2(root_node, 500)
            best_child = mcts()
            move = best_child.state.last_move
            print("AI move: ", move)

        game.make_move(move)
        root_node = Node(game)

    print(game.board)  # Display the final game state
    if game.get_winner() == 1:
        print("AI wins!")
    else:
        print("Human wins!")


if __name__ == "__main__":
    main()
