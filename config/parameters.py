'''
General configuration
'''
BOARD_SIZE = 5
TIME_LIMIT = 3
DATE = '04-19'


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
REPLAY_BUFFER_SIZE = 100
SAVE_INTERVAL = 50
NUMBER_ACTUAL_GAMES = 10000
SIMULATIONS = 10000
IDENTIFIER = 'model'
EPSILON_DECAY = 0.95
