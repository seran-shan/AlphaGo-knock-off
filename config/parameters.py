'''
General configuration
'''
BOARD_SIZE = 7
TIME_LIMIT = 2
MAX_TIME_LIMIT = 7
DATE = '04-28'
NUM_OF_MODELS = 6


'''
Configuration of neural network model
'''
INPUT_SHAPE = (BOARD_SIZE * BOARD_SIZE + 1)
OUTPUT_SHAPE = (BOARD_SIZE * BOARD_SIZE)
LAYERS = [256, 512]
ACTIVATION = 'relu'
OPTIMIZER = 'adam'
LEARNING_RATE = 1e-3
EPOCHS = 100

'''
This file contains the configuration for the reinforcement learning algorithm.
'''
REPLAY_BUFFER_SIZE = 2048
REPLAY_BATCH_SIZE = 256
SAVE_INTERVAL = 25
NUMBER_ACTUAL_GAMES = 250
SIMULATIONS = 2500
IDENTIFIER = 'model'
EPSILON_DECAY = 0.95
