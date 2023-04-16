'''
This module contains the reinforcement learning algorithm
'''
import random
from config.general import BOARD_SIZE
from config.neural_network import INPUT_SHAPE, OUTPUT_SHAPE, LAYERS, ACTIVATION, OPTIMIZER, LEARNING_RATE
from config.reinforcement_learning import NUMBER_ACTUAL_GAMES, NUMBER_SEARCH_GAMES, REPLAY_BUFFER_SIZE, SAVE_INTERVAL
from game.hex.hex import Hex
from mcts import MCTS
from mcts.node import Node
from neural_network.anet import ANet


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

    def __init__(
            self,
            anet: ANet = None,
            replay_buffer: ReplayBuffer = None,
            save_interval=None,
            number_actual_games=None,
            number_search_games=None
    ):
        self.anet = anet or None
        self.replay_buffer = replay_buffer or ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.save_interval = save_interval or SAVE_INTERVAL
        self.number_actual_games = number_actual_games or NUMBER_ACTUAL_GAMES
        self.number_search_games = number_search_games or NUMBER_SEARCH_GAMES

    def run(self, use_neural_network: bool = False):
        '''
        Run the agent
        '''
        for actual_game in range(self.number_actual_games):
            if use_neural_network:
                game = Hex(BOARD_SIZE)
                root_node = Node(game)
                mcts = MCTS(root_node, self.number_search_games, self.anet)

                while not game.is_terminal():
                    print(game.board)
                    best_child, distribution = mcts()
                    self.replay_buffer.add_case((mcts.root_node, distribution))
                    action = best_child.state.get_previous_action()
                    print("AI move: ", action)
                    mcts.root_node.state.produce_successor_state(action)
                    mcts.root_node = Node(mcts.root_node.state)

            else:
                mcts = MCTS(root_node, self.number_search_games)

                while not game.is_terminal():
                    print(game.board)
                    best_child, distribution = mcts()
                    self.replay_buffer.add_case((mcts.root_node, distribution))
                    action = best_child.state.get_previous_action()
                    print("AI move: ", action)
                    mcts.root_node.state.produce_successor_state(action)
                    mcts.root_node = Node(mcts.root_node.state)

                self.anet = ANet(
                    input_shape=INPUT_SHAPE,
                    output_shape=OUTPUT_SHAPE,
                    layers=LAYERS,
                    activation=ACTIVATION,
                    optimizer=OPTIMIZER,
                    learning_rate=LEARNING_RATE,
                )
                use_neural_network = True

            minibatch = self.replay_buffer.sample_minibatch()
            self.anet.train(minibatch)

            if actual_game % self.save_interval == 0:
                self.anet.save()
