'''
Main file for the Nim game.
'''
from config import INPUT_SHAPE, OUTPUT_SHAPE, LAYERS, ACTIVATION, OPTIMIZER, LEARNING_RATE
from game import Hex
from mcts import MCTS, Node
from neural_network import ANet
from reinforcement_learning import Agent


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
            mcts = MCTS(root_node, 500)
            best_child, dist = mcts()
            move = best_child.state.get_previous_action()
            print("AI move: ", move)
            print("Distribution: ", dist)

        game.make_move(move)
        root_node = Node(game)

    print(game.board)  # Display the final game state
    if game.get_winner() == 1:
        print("AI wins!")
    else:
        print("Human wins!")


if __name__ == "__main__":
    anet = ANet(input_shape=INPUT_SHAPE,
                output_shape=OUTPUT_SHAPE,
                layers=LAYERS,
                activation=ACTIVATION,
                optimizer=OPTIMIZER,
                learning_rate=LEARNING_RATE)
    # load_model(anet)
    agent = Agent(anet=None)
    agent.run(use_neural_network=False)
