#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:10:21 2018

@author: goncalves1
"""
import numpy as np
import sklearn.ensemble as sk
from ..base import BaseSingleEstimator


class RandomForestClassifier(BaseSingleEstimator):
    """
    Implement a wrapper for Sklearn Random Forest Classifier.

    Attributes:
        n_estimators (int): number of trees in the forest.
        criterion (str): metric to measure the quality of a split.
        max_depth (int): maximum depth of the tree.
        min_samples_split (int): minimum number of samples required to split
                    an internal node.
        min_samples_leaf (int): minimum number of samples required to be
                    at a leaf node.
    """

    def __init__(self, n_estimators=10, criterion='gini',
                 max_depth=None, min_samples_split=2,
                 min_samples_leaf=2, fit_intercept=True,
                 normalize=False, name='RF-Cl'):
        """ Initialize object with the informed hyper-parameter values.
        Args:
            n_estimators (int): number of trees in the forest.
            criterion (str): metric to measure the quality of a split.
            max_depth (int): maximum depth of the tree.
            min_samples_split (int): minimum number of samples required to
                        split an internal node.
            min_samples_leaf (int): minimum number of samples required to be
                        at a leaf node.
        """
        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        assert n_estimators >= 0
        assert min_samples_split >= 1
        assert min_samples_leaf >= 1
        assert criterion in ('gini', 'entropy')

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model = \
            sk.RandomForestClassifier(n_estimators=n_estimators,
                                      criterion=criterion,
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
        self.criterion = params['criterion']
        self.max_depth = params['max_depth']
        self.min_samples_split = params['min_samples_split']
        self.min_samples_leaf = params['min_samples_leaf']

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'n_estimators': self.n_estimators,
               'criterion': self.criterion,
               'max_depth': self.max_depth,
               'min_samples_split': self.min_samples_split,
               'min_samples_leaf': self.min_samples_leaf}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        n_estimators = [50, 100]
        criterion = ['gini', 'entropy']
        max_depth = [None, 3, 5, 10]
        min_samples_split = [2, 10]
        min_samples_leaf = [3, 5, 10]
        for r0 in n_estimators:
            for r1 in criterion:
                for r2 in max_depth:
                    for r3 in min_samples_split:
                        for r4 in min_samples_leaf:
                            yield {'n_estimators': r0,
                                   'criterion': r1,
                                   'max_depth': r2,
                                   'min_samples_split': r3,
                                   'min_samples_leaf': r4}

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (str): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())
