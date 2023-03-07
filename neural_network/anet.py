'''
This module contains a class to build a neural network model by using Keras
'''
from enum import Enum
from tensorflow import keras


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
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
            case Optimizer.ADAM:
                model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
            case Optimizer.RMSPROP:
                model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=self.learning_rate),
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
            case Optimizer.SGD:
                model.compile(optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate),
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
            case _:
                raise ValueError('Invalid optimizer')

        return model


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
