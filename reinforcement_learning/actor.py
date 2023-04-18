'''
This module contains the reinforcement learning algorithm
'''
import random
from config import *
from game.hex.hex import Hex
from mcts import MCTS
from mcts.node import Node
from neural_network.anet import ANet


class ReplayBuffer:
    '''
    Replay buffer for storing past experiences that the Actor can then use for
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


class Actor:
    '''
    Actor class
    '''

    def __init__(
            self,
            anet: ANet = None,
            replay_buffer: ReplayBuffer = None,
            save_interval=None,
            number_actual_games=None,
            simulations=None,
            identifier: str = None,
            time_limit: int = None

    ):
        self.anet = anet or None
        self.replay_buffer = replay_buffer or ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.save_interval = save_interval or SAVE_INTERVAL
        self.number_actual_games = number_actual_games or NUMBER_ACTUAL_GAMES
        self.simulations = simulations or SIMULATIONS
        self.identifier = identifier or IDENTIFIER
        self.time_limit = time_limit or TIME_LIMIT

    def run(self, use_neural_network: bool = False):
        '''
        Run the Actor
        '''
        for actual_game in range(self.number_actual_games):
            game = Hex(BOARD_SIZE)
            root_node = Node(game)
            if use_neural_network:
                mcts = MCTS(root_node, self.simulations, self.time_limit, self.anet)

                while not game.is_terminal():
                    best_child, distribution = mcts()
                    state_representation = mcts.root_node.state.extract_representation()
                    self.replay_buffer.add_case(
                        (state_representation, distribution))
                    action = best_child.state.get_previous_action()
                    print(f'\nPlayer {game.player}: {action}')
                    mcts.root_node.state.produce_successor_state(action)
                    mcts.root_node = Node(mcts.root_node.state)
                    count += 1
                    game.draw()
                print('Winner', game.get_winner())

            else:
                mcts = MCTS(root_node, self.simulations, self.time_limit)

                count = 1
                while not game.is_terminal():
                    best_child, distribution = mcts()
                    state_representation = mcts.root_node.state.extract_representation()
                    self.replay_buffer.add_case(
                        (state_representation, distribution))
                    action = best_child.state.get_previous_action()
                    print(f'\nPlayer {game.player}: {action}')
                    mcts.root_node.state.produce_successor_state(action)
                    mcts.root_node = Node(mcts.root_node.state)
                    count += 1
                    game.draw()
                print('Winner', game.get_winner())

                self.anet = ANet(
                    input_shape=INPUT_SHAPE,
                    output_shape=OUTPUT_SHAPE,
                    layers=LAYERS,
                    activation=ACTIVATION,
                    optimizer=OPTIMIZER,
                    learning_rate=LEARNING_RATE,
                )
                use_neural_network = True

            batch_size = min(REPLAY_BUFFER_SIZE, len(
                self.replay_buffer.buffer))
            minibatch = self.replay_buffer.sample_minibatch(batch_size)
            self.anet.train(minibatch)

            if actual_game % self.save_interval == 0:
                self.anet.save(self.identifier, actual_game)
