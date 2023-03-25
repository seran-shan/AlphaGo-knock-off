'''
Main file for the Nim game.
'''
from game import Nim, Hex
from mcts import MCTS, Node
from mcts.mcts2 import MCTS as MCTS2


# def main():
#     '''
#     Play Nim with human and AI.
#     '''
#     game = Nim(15, 3)
#     root_node = Node(game)

#     while not game.is_terminal():
#         print(game)  # Display the game state
#         if game.player == 0:
#             move = game.get_move()
#             print("Human move: ", move)
#         else:
#             c_punt = 1.0  # Set the exploration constant
#             mcts = MCTS(root_node, 10)
#             best_child = mcts(game.player, c_punt)
#             move = game.pieces - best_child.state.pieces
#             print("AI move: ", move)

#         game.make_move(move)
#         root_node = Node(game)

#     print(game)  # Display the final game state
#     if game.get_winner() == 1:
#         print("AI wins!")
#     else:
#         print("Human wins!")


def main(): 
    '''
    Play Hex with a human and AI.
    '''
    game = Hex(5)
    root_node = Node(game)

    while not game.is_terminal():
        print(game.board) # Display the game state
        if game.player == 0:
            move = game.get_move()
            print("Human move: ", move)
        else:
            mcts = MCTS2(root_node, 1000)
            best_child = mcts()
            move = best_child.state.last_move
            print("AI move: ", move)
        
        game.make_move(move)
        root_node = Node(game)
    
    print(game.board) # Display the final game state
    if game.get_winner() == 1:
        print("AI wins!")
    else:
        print("Human wins!")


# def main(): 
#     game = Hex(3)
#     root_node = Node(game)
#     mcts = MCTS2(root_node, 1000)
#     best_child = mcts()
#     print(best_child.state.last_move)



if __name__ == "__main__":
    main()
