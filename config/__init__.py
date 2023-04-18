'''
This module exports all the configuration parameters for the project.
'''
from .general import BOARD_SIZE, TIME_LIMIT
from .neural_network import INPUT_SHAPE, OUTPUT_SHAPE, LAYERS, ACTIVATION, OPTIMIZER, LEARNING_RATE, BATCH_SIZE
from .reinforcement_learning import NUMBER_ACTUAL_GAMES, SIMULATIONS, REPLAY_BUFFER_SIZE, SAVE_INTERVAL, IDENTIFIER
