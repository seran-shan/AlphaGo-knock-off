'''
This module contains the configuration of neural network model
'''

from enum import Enum
from config.general import BOARD_SIZE


class Optimizer(Enum):
    '''
    Optimizer enum
    '''
    ADAGRAD = 'adagrad'
    ADAM = 'adam'
    RMSPROP = 'rmsprop'
    SGD = 'sgd'


class Activation(Enum):
    '''
    Activation enum
    '''
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    SOFTMAX = 'softmax'


INPUT_SHAPE = (BOARD_SIZE * BOARD_SIZE,)
OUTPUT_SHAPE = BOARD_SIZE * BOARD_SIZE
LAYERS = [256, 256, 128]
ACTIVATION = Activation.RELU.value
OPTIMIZER = Optimizer.ADAM.value
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
