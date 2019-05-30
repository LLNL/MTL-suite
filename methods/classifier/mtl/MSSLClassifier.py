#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 08:51:38 2018

@author: goncalves1
"""
import os
import pickle
import numpy as np
import scipy.special
import scipy.optimize
from ..base import BaseMTLEstimator


def weighted_logloss(w, x, y, Omega, lambda_reg, weights):
    ''' MSSL with logloss function '''
    ntasks = Omega.shape[1]
    ndimension = int(len(w) / ntasks)
    wmat = np.reshape(w, (ndimension, ntasks), order='F')

    # make sure the data is in the correct format
    for t in range(ntasks):
        if len(y[t].shape) > 1:
            y[t] = np.squeeze(y[t])

    # cost function for each task
    cost = 0
    for t in range(ntasks):
        h_t_x = sigmoid(np.dot(x[t], wmat[:, t]))
#       h_t_x = scipy.special.expit(np.dot(x[t], wmat[:, t]))
        f1 = np.multiply(y[t], np.log(h_t_x))
        f2 = np.multiply(1 - y[t], np.log((1 - h_t_x) + 1e-8))
        f3 = np.multiply(f1 + f2, weights[t])
        cost += -f3.mean()

    # gradient of regularization term
    cost += (0.5 * lambda_reg) * np.trace(np.dot(np.dot(wmat, Omega), wmat.T))
    return cost


def weighted_logloss_der(w, x, y, Omega, lambda_reg, weights):
    ''' Gradient of the MSSL with logloss function '''

    ntasks = Omega.shape[1]
    ndimension = int(len(w) / ntasks)
    wmat = np.reshape(w, (ndimension, ntasks), order='F')

    # make sure data is in correct format
    for t in range(ntasks):
        if len(y[t].shape) > 1:
            y[t] = np.squeeze(y[t])

    # gradient of logloss term
    grad = np.zeros(wmat.shape)
    for t in range(ntasks):
        #        sig_s = scipy.special.expit(np.dot(x[t], wmat[:, t])) #[:, np.newaxis]
        sig_s = sigmoid(np.dot(x[t], wmat[:, t]))
        xweig = np.dot(x[t].T, np.diag(weights[t]))
        grad[:, t] = np.dot(xweig, (sig_s - y[t])) / x[t].shape[0]
    # gradient of regularization term
    grad += lambda_reg * np.dot(wmat, Omega)
    grad = np.reshape(grad, (ndimension * ntasks, ), order='F')
    return grad


def sigmoid(a):
    # Logit function for logistic regression
    # x: data point (array or a matrix of floats)
    # w: set of weights (array of float)

    # treating over/underflow problems
    a = np.maximum(np.minimum(a, 50), -50)
    # f = 1./(1+exp(-a));
    f = np.exp(a) / (1 + np.exp(a))
    return f


def shrinkage(a, kappa):
    return np.maximum(0, a - kappa) - np.maximum(0, -a - kappa)


class MSSLClassifier(BaseMTLEstimator):
    """
    Implement the MSSL classifier.

    Attributes:
        rho_L21 (float): l2,1 penalization hyper-parameter
        rho_L2 (float): l2 penalization hyper-parameter
    """

    def __init__(self, lambda_1=0.1, lambda_2=0,
                 fit_intercept=True, normalize=False,
                 store_params=True, name='MSSL', **kwargs):
        """ Initialize object with the informed hyper-parameter values.
        Args:
        rho_L21 (float): l2,1 penalization hyper-parameter
        rho_L2 (float): l2 penalization hyper-parameter
        """

        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        self.lambda_1 = lambda_1  # trace term
        self.lambda_2 = lambda_2  # omega sparsity
        self.max_iters = 20
        self.tol = 1e-4  # minimum tolerance: eps * 100

        self.admm_rho = 1  # 20000  # ADMM parameter
        self.eps_theta = 1e-3  # stopping criteria parameters
        self.eps_w = 1e-3  # stopping criteria parameters

        self.W = None
        self.Omega = None
        self.output_directory = ''
        self.store_params = store_params

        if 'hyper_params' in kwargs.keys():
            self.hyper_params = kwargs['hyper_params']

    def _fit(self, x, y, **kwargs):
        ''' Train MSSL model '''

        d = kwargs['censor_flag']
        v = kwargs['survival_time']

        W, Omega = self._mssl_train(x, y, d, v)
        self.W = W.copy()
        self.Omega = Omega.copy()

        if self.store_params:
            fname = os.path.join(self.output_directory, '%s.mdl' % self.__str__())
            with open(fname, 'wb') as fh:
                pickle.dump([self.W, self.Omega], fh)

    def _predict(self, x, **kwargs):
        ''' Make predictions '''

        yhat = [None] * len(x)
        for t in range(len(x)):
            yhat[t] = scipy.special.expit(np.dot(x[t], self.W[:, t]))
        return yhat

    def _mssl_train(self, x, y, d, v):

        # initialize learning parameters
        # np.linalg.solve(xmat, ymat)  #  regression warm start
        # W = -0.05 + 0.05*np.random.rand(self.nb_dims, self.nb_tasks)
        W = np.zeros((self.nb_dims, self.nb_tasks))
        Omega = np.eye(self.nb_tasks)

        weights = list()
        # compute IPCW weights
        for t in range(len(x)):
            weights.append(np.ones(x[t].shape[0]))

        # scipy opt parameters
        opts = {'maxiter': 5, 'disp': False}
        for it in range(self.max_iters):
            # print('%d ' % it)

            # Minimization step
            W_old = W.copy()
            Wvec = np.reshape(W, (self.nb_dims * self.nb_tasks, ), order='F')

#            r = scipy.optimize.check_grad(weighted_logloss,
#                                          weighted_logloss_der,
#                                          Wvec, x, y,
#                                          Omega, self.lambda_1, weights)

            additional = (x, y, Omega, self.lambda_1, weights)
            # t0 = time.time()
            res = scipy.optimize.minimize(weighted_logloss, x0=Wvec,
                                          args=additional,
                                          jac=weighted_logloss_der,
                                          method='L-BFGS-B',
                                          options=opts)
            # print('Optimization time: {}'.format(time.time() - t0))

            # put it in matrix format, where columns are coeff for each task
            W = np.reshape(res.x.copy(),
                           (self.nb_dims, self.nb_tasks), order='F')
            # Omega step:
            Omega_old = Omega

            # Learn relationship between tasks (inverse covariance matrix)
            Omega = self._omega_step(np.cov(W, rowvar=False),
                                     self.lambda_2, self.admm_rho)

            # checking convergence of Omega and W
            diff_Omega = np.linalg.norm(Omega - Omega_old)
            diff_W = np.linalg.norm(W - W_old)

            # if difference between two consecutive iterations are very small,
            # stop training
            if (diff_Omega < self.eps_theta) and (diff_W < self.eps_w):
                break

        return W, Omega

    def _omega_step(self, S, lambda_reg, rho):
        '''
        ADMM for estimation of the precision matrix.

        Input:
           S: sample covariance matrix
           lambda_reg: regularization parameter (l1-norm)
           rho: dual regularization parameter (default value = 1)
        Output:
           omega: estimated precision matrix
        '''
#        print('lambda_reg: %f'%lambda_reg)
#        print('rho: %f'%rho)
        # global constants and defaults
        max_iters = 10
        abstol = 1e-5
        reltol = 1e-5
        alpha = 1.4

        # varying penalty parameter (rho)
        mu = 10
        tau_incr = 2
        tau_decr = 2

        # get the number of dimensions
        ntasks = S.shape[0]

        # initiate primal and dual variables
        Z = np.zeros((ntasks, ntasks))
        U = np.zeros((ntasks, ntasks))

#        print('[Iters]   Primal Res.  Dual Res.')
#        print('------------------------------------')

        for k in range(0, max_iters):

            # x-update
            # numpy returns eigc_val,eig_vec as opposed to matlab's eig
            eig_val, eig_vec = np.linalg.eigh(rho * (Z - U) - S)

            # check eigenvalues
            if isinstance(eig_val[0], complex):
                print("Warning: complex eigenvalues. Check covariance matrix.")

            # eig_val is already an array (no need to get diag)
            xi = (eig_val + np.sqrt(eig_val**2 + 4 * rho)) / (2 * rho)
            X = np.dot(np.dot(eig_vec, np.diag(xi, 0)), eig_vec.T)

            # z-update with relaxation
            Zold = Z.copy()
            X_hat = alpha * X + (1 - alpha) * Zold
            Z = shrinkage(X_hat + U, lambda_reg / rho)
#            Z = Z - np.diag(np.diag(Z)) + np.eye(Z.shape[0])
            # dual variable update
            U = U + (X_hat - Z)

            # diagnostics, reporting, termination checks
            r_norm = np.linalg.norm(X - Z, 'fro')
            s_norm = np.linalg.norm(-rho * (Z - Zold), 'fro')

#            if r_norm > mu*s_norm:
#                rho = rho*tau_incr
#            elif s_norm > mu*r_norm:
#                rho = rho/tau_decr

            eps_pri = np.sqrt(ntasks**2) * abstol + reltol * max(np.linalg.norm(X, 'fro'), np.linalg.norm(Z, 'fro'))
            eps_dual = np.sqrt(ntasks**2) * abstol + reltol * np.linalg.norm(rho * U, 'fro')

            # keep track of the residuals (primal and dual)
#            print('   [%d]    %f     %f ' % (k, r_norm, s_norm))
            if r_norm < eps_pri and s_norm < eps_dual:
                break

        return Z

    def set_params(self, params):
        """
        Set hyper-parameters to be used in the execution.
        Args:
            params (dict): dict with hyper-parameter values.
        """
        self.__init__(lambda_1=params['lambda_1'],
                      lambda_2=params['lambda_2'])

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'lambda_1': self.lambda_1,
               'lambda_2': self.lambda_2}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        lambda_1 = self.hyper_params['lambda_1']
        lambda_2 = self.hyper_params['lambda_2']
        for r0 in lambda_1:
            for r1 in lambda_2:
                yield {'lambda_1': r0,
                       'lambda_2': r1}

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (str): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())

    def get_output_directory(self):
        print(self.output_directory)
