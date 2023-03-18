'''
Main file for the Nim game.
'''
from game import Nim
from mcts import MCTS, Node


def main():
    '''
    Play Nim with human and AI.
    '''
    game = Nim(15, 3)
    root_node = Node(game)

    while not game.is_terminal():
        print(game)  # Display the game state
        if game.player == 0:
            move = game.get_move()
            print("Human move: ", move)
        else:
            c_punt = 1.0  # Set the exploration constant
            mcts = MCTS(root_node, 10)
            best_child = mcts(game.player, c_punt)
            move = game.pieces - best_child.state.pieces
            print("AI move: ", move)

        game.make_move(move)
        root_node = Node(game)

    print(game)  # Display the final game state
    if game.get_winner() == 1:
        print("AI wins!")
    else:
        print("Human wins!")


if __name__ == "__main__":
    main()
