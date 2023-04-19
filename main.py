'''
This is an example of how to use the neural network actors
'''
import argparse
from neural_network import load_models
from neural_network.anet import ANet
from reinforcement_learning import Actor
from config import IDENTIFIER, BOARD_SIZE
from topp import TOPP


# def main(args):
#     '''
#     Main function

#     Parameters
#     ----------
#     args : argparse.Namespace
#         The parsed arguments
#     '''
#     if args.load_models:
#         nets = load_models(IDENTIFIER, M=3, board_size=BOARD_SIZE)
#         anet = ANet(nets[-1])
#         actor = Actor(anet=anet)
#         actor.run(use_neural_network=True)

#     elif args.tournament:
#         models = load_models(IDENTIFIER, M=2, board_size=BOARD_SIZE)
#         agents = [model for model in models]
#         tournement = TOPP(agents)
#         tournement.tournement()
#         print(tournement.results)

#     else:
#         actor = Actor(anet=None)
#         actor.run(use_neural_network=False)


# def parse_args():
#     '''
#     Parse command line arguments

#     Returns
#     -------
#     argparse.Namespace
#         The parsed arguments
#     '''
#     parser = argparse.ArgumentParser(
#         description="An example CLI for neural network actors")

#     parser.add_argument("--load_models", action="store_true",
#                         help="Load pre-trained neural network models")

#     parser.add_argument("--tournament", action="store_true",
#                         help="Run a tournament")

#     return parser.parse_args()


# if __name__ == '__main__':
#     args = parse_args()
#     main(args)
def main():
    models = load_models(IDENTIFIER,5,5)
    topp = TOPP(models)
    topp.tournement()
    for result in topp.results:
        print(result)
if __name__ == '__main__':
    main()

