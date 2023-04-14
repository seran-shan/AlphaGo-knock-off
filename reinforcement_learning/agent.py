import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from game import State
from game.hex.hex import Hex
from mcts.node import Node
from neural_network.anet import ANet
from mcts import MCTS


class ReplayBuffer:
    '''
    Replay buffer for storing past experiences that the agent can then use for
    '''

    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add_case(self, case: tuple):
        '''
        Add a case to the buffer

        Parameters
        ----------
        case : tuple
            A case to be added to the buffer
        '''
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(case)

    def sample_minibatch(self, batch_size: int):
        '''
        Sample a minibatch from the buffer

        Parameters
        ----------
        batch_size : int
            The size of the minibatch

        Returns
        -------
        list
            A minibatch of cases
        '''
        return random.sample(self.buffer, batch_size)


class Agent:
    '''
    Agent class
    '''

    # def __init__(self, anet: ANet, replay_buffer: ReplayBuffer, save_interval, number_actual_games, number_search_games):
    #     self.anet = anet
    #     self.replay_buffer = replay_buffer
    #     self.save_interval = save_interval
    #     self.number_actual_games = number_actual_games
    #     self.number_search_games = number_search_games

    def __init__(self) -> None:
        self.replay_buffer = ReplayBuffer(1000)
        self.save_interval = 100
        self.number_actual_games = 1000
        self.number_search_games = 100

    def is_final_state(self, state):
        # Implement this function according to your specific problem and design
        pass

    def backpropagate(self, mcts, final_state):
        # Implement this function according to your specific problem and design
        pass

    # def train_anet(self, minibatch):
    #     '''
    #     Train the agent's neural network

    #     Parameters
    #     ----------
    #     minibatch : list
    #         A minibatch of cases
    #     '''
    #     self.anet.train(minibatch)

    def run(self):
        '''
        Run the agent
        '''
        for actual_game in range(self.number_actual_games):
            game = Hex(5)
            root_node = Node(game)
            # FIXME: Uncomment the following line to use the MCTS with the neural network
            # mcts = MCTS(root_node, 100, self.anet)
            mcts = MCTS(root_node, 100)

            while not game.is_terminal():
                print(game.board)
                best_child, distribution = mcts()
                self.replay_buffer.add_case((mcts.root_node, distribution))
                action = best_child.state.get_previous_action()
                print("AI move: ", action)
                mcts.root_node.state.produce_successor_state(action)

            # FIXME: Uncomment the following line to use the MCTS with the neural network
            # minibatch = self.replay_buffer.sample_minibatch()
            # self.train_anet(minibatch)

            # FIXME: Uncomment the following line to use the MCTS with the neural network
            # if actual_game % self.save_interval == 0:
            #     self.anet.save()
