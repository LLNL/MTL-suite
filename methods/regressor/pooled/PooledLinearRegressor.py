#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 09:00:00 2018

@author: goncalves1
"""
import numpy as np
from sklearn import linear_model
from ..base import BaseMTLEstimator
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class PooledLinearRegressor(BaseMTLEstimator):
    """
    Implement an pooled Linear Regressor w/ L1 penalization (LASSO)
    Train one single linear model for all tasks.
    """

    def __init__(self, alpha_l1=0.5, fit_intercept=True,
                 normalize=False, name='Pooled'):
        """ Initialize object with the informed hyper-parameter values.
        Args:
            alpha_1 (float): l1 penalization hyper-parameter
        """
        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        # self.model = linear_model.SGDRegressor(loss='squared_loss',
        #                                        penalty='l1',
        #                                        alpha=alpha_l1,
        #                                        max_iter=500,
        #                                        tol=1e-4,
        #                                        fit_intercept=fit_intercept)
        self.model = linear_model.Lasso(alpha_l1)
        self.W = None
        # self.ipcw = ipcw
        self.alpha_l1 = alpha_l1

    def _fit(self, x, y, **kwargs):
        """
        Train model on given data x and y.
        Args:
            x (pandas.DataFrame): input data matrix (covariates).
            y (pandas.DataFrame): label vector (outcome).
        Returns:
            None.
        """
        # # censor flag
        # d = kwargs['censor_flag']
        # v = kwargs['survival_time']

        xpooled = np.row_stack(x)
        ypooled = np.concatenate(y)

        # print(xpooled.shape)
        # print(ypooled.shape)

        self.model.fit(xpooled, ypooled)
#        self.W = np.concatenate((self.model.coef_.ravel(), self.model.intercept_))
        self.W = self.model.coef_.ravel()
        if np.all(self.W == 0):
            warnings.warn('Pooled model: All coefficients are zero. Lower regularization parameter alpha.')
        print(self.W)

        # print('intercept: {}'.format(self.model.intercept_))
        # print(self.W)
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
            yhat[t] = self.model.predict(x[t])
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

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (string): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())
