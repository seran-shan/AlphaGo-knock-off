'''
This module contains the configuration of neural network model
'''

from enum import Enum
from config.general import BOARD_SIZE

INPUT_SHAPE = (BOARD_SIZE * BOARD_SIZE + 1)
OUTPUT_SHAPE = (BOARD_SIZE * BOARD_SIZE)
LAYERS = [64, 128]
ACTIVATION = 'relu'
OPTIMIZER = 'adam'
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 32
