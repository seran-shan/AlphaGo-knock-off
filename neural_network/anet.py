'''
This module contains a class to build a neural network model by using tf.Keras
'''
from math import sqrt
import tensorflow as tf
import numpy as np
from enum import Enum


class ANet:
    '''
    A neural network model. Implmentation of ANet is based on tf.Keras.
    '''

    def __init__(self, input_shape, output_shape, layers, activation, optimizer, learning_rate):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = layers
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model: tf.keras.Model = self.build_model()

    def build_model(self) -> tf.keras.Model:
        '''
        Build a neural network model

        Returns
        -------
        tf.keras.Model
            A neural network model
        '''
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape))
        for layer in self.layers:
            model.add(tf.keras.layers.Dense(layer, activation=self.activation))
        model.add(tf.keras.layers.Dense(
            self.output_shape, activation=Activation.SOFTMAX.value))

        match self.optimizer:
            case Optimizer.ADAGRAD.value:
                model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate),
                              loss=tf.keras.losses.BinaryCrossentropy(),
                              metrics=[tf.keras.metrics.CategoricalAccuracy()])
            case Optimizer.ADAM.value:
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                              loss=tf.keras.losses.BinaryCrossentropy(),
                              metrics=[tf.keras.metrics.CategoricalAccuracy()])
            case Optimizer.RMSPROP.value:
                model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate),
                              loss=tf.keras.losses.BinaryCrossentropy(),
                              metrics=[tf.keras.metrics.CategoricalAccuracy()])
            case Optimizer.SGD.value:
                model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
                              loss=tf.keras.losses.BinaryCrossentropy(),
                              metrics=[tf.keras.metrics.CategoricalAccuracy()])
            case _:
                raise ValueError('Invalid optimizer')

        return model

    def train(self, minibatch: list[tuple]):
        '''
        Train the neural network model

        Parameters
        ----------
        minibatch : list[tuple]
            A minibatch of cases
        '''
        feature_matrix = []
        probability_distribution = []

        for sample in minibatch:
            feature_matrix.append(sample[0])
            probability_distribution.append(sample[1])

        feature_matrix = np.array(feature_matrix)
        probability_distribution = np.array(probability_distribution)
        self.model.fit(feature_matrix, probability_distribution)

    def predict(self, node_features: np.ndarray):
        '''
        Predict the value of the state

        Parameters
        ----------
        state : numpy.ndarray
            A state of the game

        Returns
        -------
        numpy.ndarray
            A value of the state
        '''

        # input_stack = np.hstack((node_features, distribution))
        node_features = np.expand_dims(node_features, axis=0)
        return self.model.predict(node_features, verbose=0)

    def save(self, identifier: str, epoch: int):
        '''
        Save the neural network model

        Parameters
        ----------
        identifier : str
            A identifier of the model
        epoch : int
            The number of epochs
        '''
        board_shape = int(sqrt(self.output_shape))
        self.model.save(
            f'models/{identifier}_{epoch}_{board_shape}x{board_shape}')


def load_models(
    identifier: str,
    M: int,
    board_size: int,
) -> tf.keras.Model:
    '''
    Load the neural network model

    Parameters
    ----------
    identifier : str
        A identifier of the model

    M : int
        The number of models
    
    board_size : int
        The size of the board

    Returns
    -------
    tf.keras.Model
        A neural network model
    '''
    try:
       nets = [tf.keras.models.load_model(
             f'models/{identifier}_{i}_{board_size}x{board_size}') for i in range(M)]
    except OSError as exc:
        print('No model found')
    except ValueError as exc:
        print('Invalid model')
    except Exception as exc:
        print('Unexpected error')

    return nets


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
