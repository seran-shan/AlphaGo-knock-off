'''
This is an example of how to use the neural network actors
'''
import argparse

import numpy as np
from game import Hex
from mcts import Node
from neural_network import load_models
from neural_network.anet import ANet
from reinforcement_learning import Actor
from config import IDENTIFIER, BOARD_SIZE
from topp import TOPP


def main(args):
    '''
    Main function

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments
    '''
    if args.load_models:
        nets = load_models(IDENTIFIER, M=3, board_size=BOARD_SIZE)
        anet = ANet(nets[-1])
        actor = Actor(anet=anet)
        actor.run(use_neural_network=True)

    elif args.tournament:
        models = load_models(IDENTIFIER, M=6, board_size=BOARD_SIZE)
        agents = [model for model in models]
        tournement = TOPP(agents)
        tournement.tournement()
        for result in tournement.results:
            print(result)

    elif args.play:
        nets = load_models(IDENTIFIER, M=3, board_size=BOARD_SIZE)
        anet = ANet(model=nets[-1]) if nets else ANet()
        game = Hex(BOARD_SIZE)
        game.draw()
        root_node = Node(game)

        while not game.is_terminal():
            if game.player == 1:
                state_repesentation = root_node.state.extract_representation(
                    False)
                target_dist = anet.predict(state_repesentation)
                flatten_state = root_node.state.extract_flatten_state()
                legal_action = [1 if flatten_state[i] ==
                                0 else 0 for i in range(len(flatten_state))]
                target_dist = np.array(target_dist) * np.array(legal_action)
                i = np.argmax(target_dist)
                action = i // BOARD_SIZE, i % BOARD_SIZE
                print(f'\nAI move: {action}')
            else:
                action = game.get_move()
                print(f'\Your move: {action}')
            game.make_move(action)
            game.draw()
            root_node = Node(game, parent=root_node)

    else:
        actor = Actor(anet=None)
        actor.run(use_neural_network=False)


def parse_args():
    '''
    Parse command line arguments

    Returns
    -------
    argparse.Namespace
        The parsed arguments
    '''
    parser = argparse.ArgumentParser(
        description="An example CLI for neural network actors")

    parser.add_argument("--load_models", action="store_true",
                        help="Load pre-trained neural network models")

    parser.add_argument("--tournament", action="store_true",
                        help="Run a tournament")

    parser.add_argument("--play", action="store_true",
                        help="Play against the neural network model")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
