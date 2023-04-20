'''
General configuration
'''
BOARD_SIZE = 5
TIME_LIMIT = 2
DATE = '04-20'


'''
Configuration of neural network model
'''
INPUT_SHAPE = (BOARD_SIZE * BOARD_SIZE + 1)
OUTPUT_SHAPE = (BOARD_SIZE * BOARD_SIZE)
LAYERS = [256, 512]
ACTIVATION = 'relu'
OPTIMIZER = 'adam'
LEARNING_RATE = 1e-2
EPOCHS = 100
BATCH_SIZE = 32

'''
This file contains the configuration for the reinforcement learning algorithm.
'''
REPLAY_BUFFER_SIZE = 250
REPLAY_BATCH_SIZE = 32
SAVE_INTERVAL = 2
NUMBER_ACTUAL_GAMES = 10
SIMULATIONS = 500
IDENTIFIER = 'model'
EPSILON_DECAY = 0.95
