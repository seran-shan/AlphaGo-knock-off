''' 
This module contains tournament of progressive policy to determine which of 
trained agents with different evolution is the best.
'''
from config import IDENTIFIER, BOARD_SIZE
from game import Hex
from mcts import Node, MCTS
import random
import math
from neural_network import ANet, load_models
import tensorflow as tf
import numpy as np
from copy import copy


class TOPP:
    '''
    Tournament of progressive policy to determine which of trained agents 
    with different evolution is the best.

    '''

    def __init__(self, models: list[tf.keras.Model]):
        self.models = models
        self.results = []

    def play_game(self, agent1: tf.keras.Model, agent2: tf.keras.Model):
        '''
        Play a game between two agents

        Parameters
        ----------
        agent1 : ANet
            The first agent
        agent2 : ANet
            The second agent

        Returns
        -------
        int
            The winner of the game
        '''
        players = [agent1, agent2]
        player = players[0]

        game = Hex(BOARD_SIZE)
        node = Node(game)
        while not game.is_terminal():
            state_repesentation = node.state.extract_representation(False)
            target_dist = player.predict(state_repesentation, verbose=0)
            flatten_state = node.state.extract_flatten_state()
            legal_action = [1 if flatten_state[i] ==
                            0 else 0 for i in range(len(flatten_state))]
            target_dist = np.array(target_dist) * np.array(legal_action)

            i = np.argmax(target_dist)
            best_action = i // BOARD_SIZE, i % BOARD_SIZE
            node.state.produce_successor_state(best_action)

            self.change_agent(players, player)
        self.change_agent(players, player)
        self.results.append(
            f'Model {self.models.index(agent1)} vs Model {self.models.index(agent2)}:  Winner: Model {self.models.index(player)} wins')

    def tournament(self):
        '''
        Play the tournament
        '''
        for firstModel in self.models:
            models = copy(self.models)
            models.remove(firstModel)
            for secondModel in models:
                self.play_game(firstModel, secondModel)

    def change_agent(self, agents: list[ANet], agent: ANet) -> 'ANet':
        '''
        Change the agent for the next game

        Parameters
        ----------
        agent : ANet
            The agent to change

        Returns
        -------
        ANet
            The new agent
        '''
        agents = agents.copy()
        agents.remove(agent)
        return agents[0]
