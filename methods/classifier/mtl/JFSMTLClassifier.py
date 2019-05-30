#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:44:12 2018

L21 Joint Feature Learning with Logistic Loss.

OBJECTIVE
   argmin_{W,C} { sum_i^t (- sum(log (1./ (1+ exp(-X{i}*W(:, i) - Y{i} .*
                  C(i)))))/length(Y{i})) + opts.rho_L2 * \|W\|_2^2 +
                  rho1 * \|W\|_{2,1} }

INPUT:
 X: {n * d} * t - input matrix
 Y: {n * 1} * t - output matrix
 rho1: L2,1-norm group Lasso parameter.
 optional:
   opts.rho_L2: L2-norm parameter (default = 0).

OUTPUT:
 W: model: d * t
 C: model: 1 * t
 funcVal: function value vector.

RELATED PAPERS:
   [1] Evgeniou, A. and Pontil, M. Multi-task feature learning, NIPS 2007.
   [2] Liu, J. and Ye, J. Efficient L1/Lq Norm Regularization, Technical
       Report, 2010.

@author: goncalves1
"""
import os
import pickle
import numpy as np
from scipy.special import expit
from ..base import BaseMTLEstimator


class JFSMTLClassifier(BaseMTLEstimator):
    """
    Implement the L21 Joint Feature Learning classifier.

    Attributes:
        rho_L21 (float): l2,1 penalization hyper-parameter
        rho_L2 (float): l2 penalization hyper-parameter
    """

    def __init__(self, rho_L21=0.1, rho_L2=0, name='JFSMTL',
                 fit_intercept=True, normalize=False):
        """ Initialize object with the informed hyper-parameter values.
        Args:
        rho_L21 (float): l2,1 penalization hyper-parameter
        rho_L2 (float): l2 penalization hyper-parameter
        """
        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        self.rho_L21 = rho_L21
        self.rho_L2 = rho_L2
        self.max_iters = 3000
        self.tol = 1e-5  # minimum tolerance: eps * 100
        self.tFlag = 1
        self.nb_tasks = -1
        self.dimension = -1
        self.W = None
        self.C = None
        self.output_directory = ''

    def _fit(self, x, y, **kwargs):
        """
        Train model on given data x and y.
        Args:
            x (list): list of pandas.DataFrame input data matrix (covariates).
            y (list): list of pandas.DataFrame label vector (outcome).
        Returns:
            None.
        """
        self.logger.info('Traning process is about to start.')

        # get number of tasks
        self.nb_tasks = len(x)

        self.scaler = {'mu': list(), 'std': list()}
        for t in range(self.nb_tasks):
            # set-up training data - convert to numpy array
            x[t] = x[t].astype(np.float64)
            std = x[t].std(axis=0)
            std[std == 0] = 1
            self.scaler['mu'].append(x[t].mean(axis=0))
            self.scaler['std'].append(std)
            x[t] = (x[t] - x[t].mean(axis=0)) / std
            x[t] = x[t].T
            y[t] = y[t].astype(int).ravel()

        self.logger.info('Training process started.')

        self.dimension = x[0].shape[0]
        funcVal = list()

        # initialize a starting point
        W0 = np.ones((self.dimension, self.nb_tasks))
        C0 = np.ones((self.nb_tasks,))

        # this flag tests whether the gradient step only changes a little
        bFlag = False

        Wz = W0
        Cz = C0
        Wz_old = W0
        Cz_old = C0

        t = 1.0
        t_old = 0

        itrn = 0
        gamma = 1.0
        gamma_inc = 2.0
        self.logger.info('{:5} | {:13} | {:13}'.format('Iter.',
                                                       'FuncVal',
                                                       'Delta-FuncVal'))
        while itrn < self.max_iters:
            alpha = (t_old - 1) / float(t)
            Ws = (1 + alpha) * Wz - alpha * Wz_old
            Cs = (1 + alpha) * Cz - alpha * Cz_old

            # compute function value and gradients of the search point
            gWs, gCs, Fs = self.__gradVal_eval(Ws, Cs, x, y)

            # the Armijo Goldstein line search scheme
            while True:
                Wzp = self.__FGLasso_projection(Ws - gWs / gamma,
                                                self.rho_L21 / gamma)
                Czp = Cs - gCs / gamma
                Fzp = self.__funVal_eval(Wzp, Czp, x, y)

                delta_Wzp = Wzp - Ws
                delta_Czp = Czp - Cs

                nrm_delta_Wzp = np.linalg.norm(delta_Wzp, 'fro')**2
                nrm_delta_Czp = np.linalg.norm(delta_Czp)**2
                r_sum = 0.5 * (nrm_delta_Wzp + nrm_delta_Czp)

                Fzp_gamma = (Fs + np.multiply(delta_Wzp, gWs).sum()
                             + np.multiply(delta_Czp, gCs).sum()
                             + gamma / 2.0 * nrm_delta_Wzp
                             + gamma / 2.0 * nrm_delta_Czp)

                if r_sum <= 1e-20:
                    # gradient step makes little improvement
                    bFlag = True
                    break
                if Fzp <= Fzp_gamma:
                    break
                else:
                    gamma *= gamma_inc

            Wz_old = Wz
            Cz_old = Cz
            Wz = Wzp
            Cz = Czp

            funcVal.append(Fzp + self.__nonsmooth_eval(Wz, self.rho_L21))

            if itrn > 1:
                self.logger.info('{:^5} | {} | {}'.format(itrn,
                                                          funcVal[-1],
                                                          abs(funcVal[-1] -
                                                              funcVal[-2])))

            if bFlag:
                # The program terminates as the gradient step
                # changes the solution very small
                self.logger.info(('The program terminates as the gradient step'
                                  'changes the solution very small'))
                break

            # test stop condition.
            if self.tFlag == 0:
                if itrn >= 2:
                    if abs(funcVal[-1] - funcVal[-2]) <= self.tol:
                        break
            elif self.tFlag == 1:
                if itrn >= 2:
                    if (abs(funcVal[-1] - funcVal[-2]) <
                            (self.tol * funcVal[-2])):
                        break
            elif self.tFlag == 2:
                if funcVal[-1] <= self.tol:
                    break
            elif self.tFlag == 3:
                if itrn >= self.max_iters:
                    break
            else:
                raise ValueError('Unknown termination flag')

            itrn = itrn + 1
            t_old = t
            t = 0.5 * (1 + (1 + 4 * t**2)**0.5)

        self.W = Wzp
        self.C = Czp

        # save model into pickle file
        filename = '{}.model'.format(self.__str__())
        filename = os.path.join(self.output_directory, filename)
        with open(filename, "wb") as fh:
            pickle.dump([self.W, self.C], fh)

    def _predict(self, x, **kwargs):
        """ Predict regression value for the input x.
        Args:
            x (pandas.DataFrame): list of pandas.DataFrame input data matrix.
        Returns:
            list of numpy.array with the predicted values.
        """
        y_hats = list()
        for t in range(self.nb_tasks):
            # convert to numpy array
            x[t] = x[t].astype(np.float64)
            x[t] = (x[t] - self.scaler['mu'][t]) / self.scaler['std'][t]
            y_hat = expit(np.dot(x[t], self.W[:, t]) + self.C[t])
            y_hat = np.around(y_hat).astype(int)
            y_hats.append(y_hat)
        return y_hats

    def set_params(self, params):
        """
        Set hyper-parameters to be used in the execution.
        Args:
            params (dict): dict with hyper-parameter values.
        """
        self.rho_L21 = params['rho_L21']
        self.rho_L2 = params['rho_L2']

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'rho_L21': self.rho_L21,
               'rho_L2': self.rho_L2}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        rho_L21 = np.logspace(-10, 1, 10)
        rho_L2 = np.logspace(-10, 1, 10)
        for r0 in rho_L21:
            for r1 in rho_L2:
                yield {'rho_L21': r0,
                       'rho_L2': r1}

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (str): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())

    def __FGLasso_projection(self, W, rho):
        """ Solve it in row wise (L_{2,1} is row coupled).
         for each row we need to solve the proximal operator
         argmin_w { 0.5 \|w - v\|_2^2 + lambda_3 * \|w\|_2 }
        """
        Wp = np.zeros(W.shape)
        for i in range(W.shape[0]):
            nm = np.linalg.norm(W[i, :], 2)
            if nm == 0:
                w = np.zeros((W.shape[1], 1))
            else:
                w = max(nm - rho, 0) / nm * W[i, :]
            Wp[i, :] = w.T
        return Wp

    def __gradVal_eval(self, W, C, X, Y):
        """smooth part gradient
        """
        grad_W = np.zeros((self.dimension, self.nb_tasks))
        grad_C = np.zeros((self.nb_tasks,))
        lossValVect = np.zeros((self.nb_tasks,))
        for t in range(self.nb_tasks):
            grad_W[:, t], grad_C[t], lossValVect[t] = \
                self.__unit_grad_eval(W[:, t], C[t], X[t], Y[t])
        grad_W += self.rho_L2 * 2 * W
        # when computing function value we do not include l1 norm.
        funcVal = lossValVect.sum() + self.rho_L2 * np.linalg.norm(W, 'fro')**2
        return grad_W, grad_C, funcVal

    def __unit_grad_eval(self, w, c, x, y):
        """ gradient and logistic evaluation for each task """
        m = max(y.shape)
        weight = np.ones((m,)) / float(m)
        weighty = np.multiply(weight, y)
        aa = np.multiply(-y, np.dot(x.T, w) + c)
        bb = np.maximum(aa, 0)
        funcVal = np.dot(weight.T, (np.log(np.exp(-bb) + np.exp(aa - bb)) + bb))
        pp = 1.0 / (1 + np.exp(aa))
        b = np.multiply(-weighty, (1 - pp))
        grad_c = sum(b)
        grad_w = np.dot(x, b)
        return grad_w, grad_c, funcVal

    def __unit_funcVal_eval(self, w, c, x, y):
        """ function value evaluation for each task"""
        m = max(y.shape)
        weight = np.ones((m,)) / float(m)
        aa = np.multiply(-y, (np.dot(x.T, w) + c))
        bb = np.maximum(aa, 0)
        return np.dot(weight.T, np.log(np.exp(-bb) + np.exp(aa - bb)) + bb)

    def __funVal_eval(self, W, C, X, Y):
        """ smooth part function value."""
        funcVal = 0
        for t in range(self.nb_tasks):
            funcVal += self.__unit_funcVal_eval(W[:, t], C[t], X[t], Y[t])
        funcVal += self.rho_L2 * np.linalg.norm(W, 'fro')**2
        return funcVal

    def __nonsmooth_eval(self, W, rho_1):
        """ non-smooth part function valeu. """
        non_smooth_value = 0
        for i in range(self.dimension):
            non_smooth_value += rho_1 * np.linalg.norm(W[i, :], 2)
        return non_smooth_value
