#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:57:09 2018

@author: goncalves1
"""
import os
import keras.callbacks as cllbck
from keras.models import Model
import keras.layers as layers
from ..base import BaseSingleEstimator


class DNNRegressor(BaseSingleEstimator):
    """
    Implement a Deep-and-Wide Multi-layer Perceptron Regressor with Dropout and
    BatchNormalization for overfitting avoidance and training stability.

    Attributes:
        batch_size (int): size of the mini-batch.
        nb_epochs (int): number of epochs for training
    """

    def __init__(self, batch_size=100, nb_epochs=100, arch=[10, 10],
                 fit_intercept=True, normalize=False, name='DNNR-STL'):
        """ Initialize object with the informed hyper-parameter values.
        Args:
            batch_size (int): mini-batch size.
            nb_epochs (int): number of training epochs.
        """
        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        assert batch_size >= 0
        assert nb_epochs >= 0

        self.arch = arch
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs

        self.model = None

    def _fit(self, x, y, **kwargs):
        """
        Train model on given data x and y.
        Args:
            x (numpy.array): input data matrix (covariates).
            y (numpy.array): label vector (outcome).
        Returns:
            None.
        """
        self.logger.info('Traning process is about to start.')

        self.ndimensions = x.shape[1]  # dimension of the data

        # architecture depend on data dimension
        self._create_model((x.shape[1],))
        self.logger.info('Model architecture created.')

        monitor = 'val_loss'

        filename = '{}.hdf5'.format(self.__str__())
        filename = os.path.join(self.output_directory, filename)

        # callbacks
        early_stopping = cllbck.EarlyStopping(monitor=monitor, patience=10,
                                              verbose=0)
        checkpoint = cllbck.ModelCheckpoint(filename, monitor=monitor,
                                            verbose=0, save_best_only=True)
#        tbCallBack = cllbck.TensorBoard(log_dir='./log', histogram_freq=0,
#                                        write_graph=True, write_images=True)
#        reduce_lr = cllbck.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                                             patience=5, min_lr=0.001)
        callbacks_list = [early_stopping, checkpoint]  # , reduce_lr]  # tbCallBack

        self.logger.info('Training process started.')

        # train NN model on input data
        self.model.fit(x, y, epochs=self.nb_epochs, verbose=0,
                       batch_size=self.batch_size, callbacks=callbacks_list,
                       validation_split=0.2)

        # load best model' weights from file
        self.model.load_weights(filename)
        self.logger.info('Training process finalized.')

    def _predict(self, x, **kwargs):
        """ Predict regression value for the input x.
        Args:
            x (pandas.DataFrame): input data matrix.
        Returns:
            numpy.array with the predicted values.
        """
        # convert to numpy array
        return self.model.predict(x).ravel()

    def set_params(self, params):
        """
        Set hyper-parameters to be used in the execution.
        Args:
            params (dict): dict with hyper-parameter values.
        """
        self.batch_size = params['batch_size']

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'batch_size': self.batch_size}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        batch_sizes = [16, 32, 64, 100, 200]
        for bt_size in batch_sizes:
            yield {'batch_size': bt_size}

    def _create_model(self, input_shape):
        """ Multilayer perceptron with Batch Normalization and Dropout. """

        # input (feed in) layer
        inputs = layers.Input(shape=input_shape)

        # to be connected to the final layer
        fc = inputs

        # Fully connected layers - 1st hidden layer
        for ll in self.arch:
            fc = layers.Dense(ll)(fc)
            fc = layers.BatchNormalization()(fc)
            fc = layers.Activation('sigmoid')(fc)

        # deep and wide model
        cat1 = layers.concatenate([fc, inputs], axis=-1)
        cat1 = layers.BatchNormalization()(cat1)

        # Output prediction
        out = layers.Dense(1, activation='linear')(cat1)
        self.model = Model(inputs, out, name='mlp')

        self.model.compile(optimizer='rmsprop',
                           loss='mean_squared_error',
                           metrics=['mean_squared_error'])
