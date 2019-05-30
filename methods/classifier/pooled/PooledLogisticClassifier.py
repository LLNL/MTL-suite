#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 08:59:14 2018

@author: goncalves1
"""
import numpy as np
from sklearn import linear_model
from ..base import BaseMTLEstimator


class PooledLogisticClassifier(BaseMTLEstimator):
    """
    Implement an pooled Logistic Classifier: train one single logistic
    classifier for all tasks.
    """

    def __init__(self, alpha_l1=0.5, fit_intercept=True,
                 normalize=False, name='Pooled-LR'):
        """ Initialize object with the informed hyper-parameter values.
        Args:
            alpha_1 (float): l1 penalization hyper-parameter
        """
        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        self.model = linear_model.LogisticRegression(penalty='l1',
                                                     C=alpha_l1,
                                                     solver='liblinear')
        self.alpha_l1 = alpha_l1
        self.W = None

    def _fit(self, x, y, **kwargs):
        """
        Train model on given data x and y.
        Args:
            x (pandas.DataFrame): input data matrix (covariates).
            y (pandas.DataFrame): label vector (outcome).
        Returns:
            None.
        """
        # self.logger.info('Traning process is about to start.')

        self.ntasks = len(x)  # get number of tasks
        self.dimension = x[0].shape[1]  # get problem dimension

        for t in range(self.ntasks):
            x[t] = x[t].astype(np.float64)
            y[t] = y[t].astype(np.int32).ravel()

        xpooled = np.row_stack(x)
        ypooled = np.concatenate(y)

        # Train the model using the training sets
        self.model.fit(xpooled, ypooled)
        self.W = np.concatenate((self.model.coef_.ravel(), self.model.intercept_))

        # self.logger.info('Training process finalized.')

    def _predict(self, x, **kwargs):
        """ Predict regression value for the input x.
        Args:
            x (numpy.array): input data matrix.
        Returns:
            numpy.array with the predicted values.
        """
        yhat = [None] * len(x)
        for t in range(len(x)):
            x[t] = x[t].astype(np.float64)
            yhat[t] = np.around(self.model.predict(x[t])).astype(int)
        return yhat

    def set_params(self, params):
        """
        Set hyper-parameters to be used in the execution.
        Args:
            params (dict): dict with hyper-parameter values.
        """
        self.alpha_l1 = params['alpha_l1']

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'alpha_l1': self.alpha_l1}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        alphas_l1 = np.logspace(-6, 3, 50)
        for alpha_l1 in alphas_l1:
            yield {'alpha_l1': alpha_l1}

    # def set_output_directory(self, output_dir):
    #     """ Set output folder path.
    #     Args:
    #         output_dir (string): path to output directory.
    #     """
    #     self.output_directory = output_dir
    #     self.logger.set_path(output_dir)
    #     self.logger.setup_logger(self.__str__())
