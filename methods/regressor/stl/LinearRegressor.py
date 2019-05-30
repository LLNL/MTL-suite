#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:33:16 2018

@author: goncalves1
"""
import sys
import os
import pickle
import numpy as np
from sklearn import linear_model
from ..base import BaseSingleEstimator


class LinearRegressor(BaseSingleEstimator):
    """
    Implement an Ordinary Linear Regression model (non-regularized) using
    scikit-learn.
    """

    def __init__(self, alpha=1.0, name='Lin-Reg',
                 fit_intercept=True, normalize=False):
        """ Initialize object with the informed hyper-parameter values.
        Args:
            batch_size (int): mini-batch size.
            nb_epochs (int): number of training epochs.
        """
        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        self.alpha_l1 = alpha
        self.model = linear_model.Lasso(self.alpha_l1)

    def _fit(self, x, y, **kwargs):
        """
        Train model on given data x and y.
        Args:
            x (pandas.DataFrame): input data matrix (covariates).
            y (pandas.DataFrame): label vector (outcome).
        Returns:
            None.
        """
        # task_id = kwargs['task_id']
        self.model.alpha = self.alpha_l1
        self.model.fit(x, y)

        fname = os.path.join(self.output_directory,
                             '{}_T{}.params'.format(self.__str__(), kwargs['task_id']))
        with open(fname, 'wb') as fh:
            pickle.dump(self.model.coef_, fh)

        #self.logger.info('Training process finalized.')

    def _predict(self, x):
        """ Predict regression value for the input x.
        Args:
            x (numpy.array): input data matrix.
        Returns:
            numpy.array with the predicted values.
        """
        return self.model.predict(x)

    def set_params(self, params):
        """
        Set hyper-parameters to be used in the execution.
        Args:
            params (dict): dict with hyper-parameter values.
        """
        self.alpha_l1 = params['alpha_l1']
        # self.__init__(name=self.name, alpha=self.alpha_l1)

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'alpha_l1': self.alpha_l1}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        alphas_l1 = np.logspace(-4, 2, 30)
        # alphas_l1 = np.linspace(1e-4, 1e3, 20)
        for alpha_l1 in alphas_l1:
            yield {'alpha_l1': alpha_l1}

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (string): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())
