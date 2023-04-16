'''
This file contains the configuration for the reinforcement learning algorithm.
'''
from reinforcement_learning.agent import ReplayBuffer

REPLAY_BUFFER_SIZE = 100000
REPLAY_BUFFER = ReplayBuffer(REPLAY_BUFFER_SIZE)
SAVE_INTERVAL = 100
NUMBER_ACTUAL_GAMES = 1000
NUMBER_SEARCH_GAMES = 100
