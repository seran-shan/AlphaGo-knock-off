'''
This module contains a class to build a neural network model by using Keras
'''
from enum import Enum
from tensorflow import keras
import numpy as np


class ANet:
    '''
    A neural network model. Implmentation of ANet is based on Keras.
    '''

    def __init__(self, input_shape, output_shape, layers, activation, optimizer, learning_rate):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = layers
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self) -> keras.Model:
        '''
        Build a neural network model

        Returns
        -------
        keras.Model
            A neural network model
        '''
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=self.input_shape))
        for layer in self.layers:
            model.add(keras.layers.Dense(layer, activation=self.activation))
        model.add(keras.layers.Dense(
            self.output_shape, activation=Activation.SOFTMAX))

        match self.optimizer:
            case Optimizer.ADAGRAD:
                model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=self.learning_rate),
                              loss=keras.losses.SparseCategoricalCrossentropy(),
                              metrics=[keras.metrics.SparseCategoricalAccuracy()])
            case Optimizer.ADAM:
                model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                              loss=keras.losses.SparseCategoricalCrossentropy(),
                              metrics=[keras.metrics.SparseCategoricalAccuracy()])
            case Optimizer.RMSPROP:
                model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=self.learning_rate),
                              loss=keras.losses.SparseCategoricalCrossentropy(),
                              metrics=[keras.metrics.SparseCategoricalAccuracy()])
            case Optimizer.SGD:
                model.compile(optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate),
                              loss=keras.losses.SparseCategoricalCrossentropy(),
                              metrics=[keras.metrics.SparseCategoricalAccuracy()])
            case _:
                raise ValueError('Invalid optimizer')

        return model

    def train(self, state: np.ndarray, target: np.ndarray, batch_size=1):
        '''
        Train the neural network model

        Parameters
        ----------
        state : numpy.ndarray
            A state of the game
        target : numpy.ndarray
            A target value of the state
        batch_size : int
            The number of samples per gradient update
        '''
        self.model.fit(state, target, batch_size=batch_size)

    def predict(self, state):
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
        return self.model.predict(state)

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
        self.model.save(f'models/{identifier}_{epoch}.h5')


def load_model(identifier: str, M: int) -> keras.Model:
    '''
    Load the neural network model

    Parameters
    ----------
    identifier : str
        A identifier of the model

    M : int
        The number of models

    Returns
    -------
    keras.Model
        A neural network model
    '''
    try:
        nets = [keras.models.load_model(
            f'models/{identifier}_{i + 1}.h5') for i in range(M)]
    except OSError as exc:
        raise OSError('No model found') from exc
    except ValueError as exc:
        raise ValueError('Invalid model') from exc
    except Exception as exc:
        raise Exception('Unknown error') from exc

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
