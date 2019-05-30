#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:24:42 2018

@author: widemann1
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as gbr
from ..base import BaseSingleEstimator
# from design import Method


class GradientBoostingRegressor(BaseSingleEstimator):
    """
    Implement a wrapper for xgboost.

    Attributes:

    """

    def __init__(self, n_estimators=100, max_depth=2, learning_rate=0.01,
                 loss='ls', max_features=None, min_samples_split=2,
                 min_samples_leaf=3, alpha=.09, verbose=0, name='GBR-Reg',
                 fit_intercept=True, normalize=False):
        """ Initialize object with the informed hyper-parameter values.
        Args:

(self, lambda_1=0.1, lambda_2=0, name='MSSL',
                 fit_intercept=True, normalize=False)

        """
        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.min_samples_leaf = min_samples_leaf
        self.output_directory = ''
        self.model = gbr(n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         max_features=max_features,
                         verbose=verbose,
                         loss=loss)

    def _fit(self, x, y, **kwargs):
        """
        Train model on given data x and y.
        Args:
            x (pandas.DataFrame): input data matrix (covariates).
            y (pandas.DataFrame): label vector (outcome).
        Returns:
            None.
        """
        self.logger.info('Traning process is about to start.')

        # extract covariate names for later use
        # column_names = kwargs['column_names']

        # set-up training data - convert to numpy array
        x = x.astype(np.float64)
        y = y.astype(np.float64).ravel()

        self.logger.info('Training process started.')
        self.model.fit(x, y)

        # # get variable importance computed based on training
        # feature_importance = self.model.feature_importances_
        # z = zip(feature_importance, column_names)
        # z = sorted(z, key=lambda x_i: x_i[0], reverse=True)
        # for f in z:
        #     self.logger.info('feature: {}: {:.8f}'.format(f[1], f[0]))

        # self.logger.info('Training process finalized.')

    def _predict(self, x, **kwargs):
        """ Predict regression value for the input x.
        Args:
            x (pandas.DataFrame): input data matrix.
        Returns:
            numpy.array with the predicted values.
        """
        # convert to numpy array
        x = x.astype(np.float64)

        return self.model.predict(x)

    def set_params(self, params):
        """
        Set hyper-parameters to be used in the execution.
        Args:
            params (dict): dict with hyper-parameter values.
        """
        self.n_estimators = params['n_estimators']
        self.max_depth = params['max_depth']
        self.min_samples_split = params['min_samples_split']
        self.min_samples_leaf = params['min_samples_leaf']
        self.loss = params['loss']
#        self.learning_rate = params['learning_rate']

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'n_estimators': self.n_estimators,
               'max_depth': self.max_depth,
               'min_samples_split': self.min_samples_split,
               'min_samples_leaf': self.min_samples_leaf,
               'loss': self.loss,
               }
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        n_estimators = [50, 100]
        max_depth = [None, 5, 10]
        min_samples_split = [2, 10]
        min_samples_leaf = [3, 10]
        losses = ['ls', 'lad']
        for r0 in n_estimators:
            for r1 in max_depth:
                for r2 in min_samples_split:
                    for r3 in min_samples_leaf:
                        for r4 in losses:
                            yield {'n_estimators': r0,
                                   'max_depth': r1,
                                   'min_samples_split': r2,
                                   'min_samples_leaf': r3,
                                   'loss': r4}

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (str): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())
