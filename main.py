'''
This is an example of how to use the neural network actors
'''
import argparse
from neural_network import load_models
from reinforcement_learning import Actor
from config import IDENTIFIER


def main(args):
    '''
    Main function

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments
    '''
    if args.load_models:
        nets = load_models(IDENTIFIER, M=11, board_size=4, save_interval=5)
        actor = Actor(anet=nets[1])
        actor.run(use_neural_network=True)

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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
