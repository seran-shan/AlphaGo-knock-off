'''
This is an example of how to use the neural network agents
'''
import argparse
from neural_network import load_models
from reinforcement_learning import Agent


def main(args):
    '''
    Main function

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments
    '''
    if args.load_models:
        nets = load_models('models', 1)
        agent = Agent(anet=nets[0])
        agent.run(use_neural_network=True)

    else:
        agent = Agent(anet=None)
        agent.run(use_neural_network=False)


def parse_args():
    '''
    Parse command line arguments

    Returns
    -------
    argparse.Namespace
        The parsed arguments
    '''
    parser = argparse.ArgumentParser(
        description="An example CLI for neural network agents")

    parser.add_argument("--load_models", action="store_true",
                        help="Load pre-trained neural network models")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
