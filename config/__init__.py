'''
This module exports all the configuration parameters for the project.
'''
from .general import BOARD_SIZE
from .neural_network import INPUT_SHAPE, OUTPUT_SHAPE, LAYERS, ACTIVATION, OPTIMIZER, LEARNING_RATE, BATCH_SIZE, Activation, Optimizer
from .reinforcement_learning import NUMBER_ACTUAL_GAMES, NUMBER_SEARCH_GAMES, REPLAY_BUFFER_SIZE, SAVE_INTERVAL
