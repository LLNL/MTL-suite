#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:50:33 2018

@author: goncalves1
"""
import numpy as np
import sklearn.ensemble as sk
from ..base import BaseMTLEstimator
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class PooledRandomForestClassifier(BaseMTLEstimator):
    """
    Implement a wrapper for Sklearn Random Forest Regressor.

    Attributes:
        n_estimators (int): number of trees in the forest.
        max_depth (int): maximum depth of the tree.
        min_samples_split (int): minimum number of samples required to split
                    an internal node.
        min_samples_leaf (int): minimum number of samples required to be
                    at a leaf node.
    """

    def __init__(self, n_estimators=10, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, fit_intercept=True,
                 normalize=False, name='RFC-Pooled'):
        """ Initialize object with the informed hyper-parameter values.
        Args:
        n_estimators (int): number of trees in the forest.
        max_depth (int): maximum depth of the tree.
        min_samples_split (int): minimum number of samples required to split
                    an internal node.
        min_samples_leaf (int): minimum number of samples required to be
                    at a leaf node.
        """
        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        assert n_estimators >= 0
        assert min_samples_split >= 1
        assert min_samples_leaf >= 1

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model = sk.RandomForestClassifier(n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               min_samples_split=min_samples_split,
                                               min_samples_leaf=min_samples_leaf,
                                               n_jobs=10)
        self.output_directory = ''

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
        # get number of tasks
        self.ntasks = len(x)
        self.dimension = x[0].shape[1]

        # extract covariate names for later use
        # column_names = kwargs['column_names']

        for t in range(self.ntasks):
            x[t] = x[t].astype(np.float64)
            y[t] = y[t].astype(np.float64).ravel()

        xpooled = np.row_stack(x)
        ypooled = np.concatenate(y)

        self.model.fit(xpooled, ypooled)

        # get variable importance computed based on training
        # feature_importance = self.model.feature_importances_
        # print(sorted(feature_importance))
        # z = zip(feature_importance, column_names)
        # z = sorted(z, key=lambda x_i: x_i[0], reverse=True)
        # for f in z:
        #     self.logger.info('feature: {}: {:.8f}'.format(f[1], f[0]))

        self.logger.info('Training process finalized.')

    def _predict(self, x, **kwargs):
        """ Predict regression value for the input x.
        Args:
            x (pandas.DataFrame): input data matrix.
        Returns:
            numpy.array with the predicted values.
        """
        yhat = [None] * len(x)
        for t in range(len(x)):
            x[t] = x[t].astype(np.float64)
            yhat[t] = self.model.predict(x[t])
            yhat[t] = np.maximum(0, yhat[t])
        return yhat

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

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'n_estimators': self.n_estimators,
               'max_depth': self.max_depth,
               'min_samples_split': self.min_samples_split,
               'min_samples_leaf': self.min_samples_leaf}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        n_estimators = [10, 50, 100]
        max_depth = [None, 3, 5, 10]
        min_samples_split = [2, 10]
        min_samples_leaf = [3, 5, 10]
        for r0 in n_estimators:
            for r1 in max_depth:
                for r2 in min_samples_split:
                    for r3 in min_samples_leaf:
                        yield {'n_estimators': r0,
                               'max_depth': r1,
                               'min_samples_split': r2,
                               'min_samples_leaf': r3}

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (str): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())
