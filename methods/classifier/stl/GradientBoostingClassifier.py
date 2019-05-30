#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:23:13 2018

@author: goncalves1
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as gbr
from ..base import BaseSingleEstimator


class GradientBoostingClassifier(BaseSingleEstimator):
    """
    Implement a wrapper for sklearn GradientBoostingClassifier.

    Attributes:

    """

    def __init__(self, n_estimators=100, max_depth=10, learning_rate=0.01,
                 loss='deviance', max_features=None, min_samples_split=2,
                 min_samples_leaf=1, verbose=0, fit_intercept=True,
                 normalize=False, name='GBC'):
        """ Initialize object with the informed hyper-parameter values.
        Args:

        """
        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        assert loss in ('deviance', 'exponential')
        assert n_estimators >= 1
        assert max_depth >= 2
        assert learning_rate > 0
        assert min_samples_split >= 1
        assert min_samples_leaf >= 1

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
        column_names = kwargs['column_names']

        # set-up training data - convert to numpy array
        x = x.astype(np.float64)
        y = y.astype(int).ravel()

        self.logger.info('Training process started.')
        self.model.fit(x, y)

        # get variable importance computed based on training
        feature_importance = self.model.feature_importances_
        z = zip(feature_importance, column_names)
        z = sorted(z, key=lambda x_i: x_i[0], reverse=True)
        for f in z:
            self.logger.info('feature: {}: {:.8f}'.format(f[1], f[0]))

        self.logger.info('Training process finalized.')

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
               'loss': self.loss,
               'max_depth': self.max_depth,
               'min_samples_split': self.min_samples_split,
               'min_samples_leaf': self.min_samples_leaf}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        n_estimators = [50, 100]
        losses = ['deviance', 'exponential']
        max_depth = [None, 3, 5, 10]
        min_samples_split = [2, 10]
        min_samples_leaf = [3, 5, 10]
        for r0 in n_estimators:
            for r1 in losses:
                for r2 in max_depth:
                    for r3 in min_samples_split:
                        for r4 in min_samples_leaf:
                            yield {'n_estimators': r0,
                                   'loss': r1,
                                   'max_depth': r2,
                                   'min_samples_split': r3,
                                   'min_samples_leaf': r4}
