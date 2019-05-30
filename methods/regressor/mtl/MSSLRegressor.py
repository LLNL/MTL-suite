#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 08:46:32 2018

@author: goncalves1
"""
import sys
import pickle
import numpy as np
import scipy.special
import scipy.optimize
import scipy.sparse
import scipy.linalg
import os
from ..base import BaseMTLEstimator


class MSSLRegressor(BaseMTLEstimator):
    """ Implements the MSSL regressor. """

    def __init__(self, lambda_1=0.1, lambda_2=0, name='MSSL',
                 fit_intercept=True, normalize=False):
        """ Initialize object with the informed hyper-parameter values.
        Args:
            lambda_1 (float): Omega penalization hyper-parameter
            lambda_2 (float): W l1-penalization hyper-parameter
        """
        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        self.name = name
        self.wstep_alg = 'gradient-based'  # or 'closed-form'

        self.lambda_1 = lambda_1  # trace term
        self.lambda_2 = lambda_2  # omega sparsity

        self.W = None  # tasks coefficients
        self.Omega = None  # tasks dependence matrix

        self.max_iters = 100
        self.tol = 1e-5  # minimum tolerance: eps * 100
        self.sparse = False
        self.admm_rho = 1  # ADMM parameter
        self.eps_theta = 1e-3  # stopping criteria parameters
        self.eps_w = 1e-3  # stopping criteria parameters

    def _fit(self, x, y, **kwargs):

        self.W, self.Omega = self._mssl_train(x, y)

        # store learned parameters
        fname = os.path.join(self.output_directory,
                             '{}.params'.format(self.__str__()))
        with open(fname, 'wb') as fh:
            pickle.dump([self.W, self.Omega], fh)

    def _predict(self, x):

        yhat = [None] * len(x)
        for t in range(len(x)):
            yhat[t] = np.dot(x[t], self.W[:, t])
        return yhat

    def _mssl_train(self, x, y):

        if self.wstep_alg == 'closed-form':
            # create permutation matrix
            P = self.create_permutation_matrix(self.nb_dims, self.nb_tasks)

            # create (sparse) matrices x and y that are the concatenation
            # of data from all tasks. So, in this way we convert multiple
            # tasks problem to a single (bigger) problem used in the
            # closed-form solution. For gradient-based algorithms like Fista,
            # L-BFGS and others, this is not needed (although can also be used)
            xmat, ymat = self.create_sparse_AC(x, y)

            # initialize learning parameters
            Wvec = scipy.linalg.lstsq(xmat, ymat)  # regression warm start
        else:
            # initialize learning parameters
            W = -0.05 + 0.05 * np.random.rand(self.nb_dims, self.nb_tasks)

        Omega = np.eye(self.nb_tasks)

        # scipy opt parameters
        for it in range(self.max_iters):

            # Minimization step
            W_old = W.copy()
            Wvec = np.reshape(W, (self.nb_dims * self.nb_tasks, ), order='F')

            if self.wstep_alg == 'closed-form':
                factor = self.lambda_1
                # TODO: weighted version
                # https://stackoverflow.com/questions/27128688/how-to-use-least-squares-with-weight-matrix-in-python
                #W = [1,2,3,4,5]
                #W = np.sqrt(np.diag(W))
                #Aw = np.dot(W,A)
                #Bw = np.dot(B,W)
                L = np.kron(np.eye(self.nb_dims), Omega)
                Xls = xmat + np.dot(np.dot(factor * P, L), P.T)
#                if Xls.shape[0] == Xls.shape[1]:  # squared Xls
#                    Wvec = np.linalg.solve(Xls, ymat)
#                else:
                Wvec, _, _, _ = scipy.linalg.lstsq(Xls, ymat)

            elif self.wstep_alg == 'gradient-based':
                opts = {'maxiter': 10, 'disp': False}
                # r = scipy.optimize.check_grad(squaredloss,
                #                              squaredloss_der,
                #                              Wvec, x, y, Omega, self.lambda_1)
                additional = (x, y, Omega, self.lambda_1)
                res = scipy.optimize.minimize(squaredloss, x0=Wvec,
                                              args=additional,
                                              jac=squaredloss_der,
                                              method='BFGS', options=opts)
                Wvec = res.x.copy()
            else:
                raise NotImplementedError("W-step {} not found".format(self.wstep_alg))

            # put it in matrix format, where columns are coeff for each task
            W = np.reshape(Wvec, (self.nb_dims, self.nb_tasks), order='F')

            # Omega step:
            Omega_old = Omega.copy()

            # Learn relationship between tasks (inverse covariance matrix)
            Omega = self._omega_step(np.cov(W, rowvar=False),
                                     self.lambda_2, self.admm_rho)

            # checking convergence of Omega and W
            diff_Omega = np.linalg.norm(Omega - Omega_old, 'fro')
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

        # global constants and defaults
        max_iters = 10
        abstol = 1e-5
        reltol = 1e-5
        alpha = 1.4

        if len(S.shape) == 0:
            ntasks = 1
        else:
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

            # dual variable update
            U = U + (X_hat - Z)

            # diagnostics, reporting, termination checks
            r_norm = np.linalg.norm(X - Z, 'fro')
            s_norm = np.linalg.norm(-rho * (Z - Zold), 'fro')

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
        self.lambda_1 = params['lambda_1']
        self.lambda_2 = params['lambda_2']

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
        lambda_1 = np.logspace(-7, 2, 10)
        lambda_2 = np.logspace(0, 4, 10)
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

    def create_permutation_matrix(self, k, v):
        if self.sparse:
            Imat = scipy.sparse.eye(k * v)
            permV = scipy.sparse.csr_matrix((k * v + 1, 1))
        else:
            Imat = np.eye(k * v)
            permV = np.zeros((k * v + 1, 1), dtype=int)
        cur = 1
        for i in range(1, v + 1):
            for j in range(1, k + 1):
                permV[cur, 0] = i + ((j - 1) * v)
                cur = cur + 1
        permV = permV[1:] - 1
        p = Imat[:, permV].T
        return np.squeeze(p)

    def create_sparse_AC(self, A, b):
        nmodels = A[0].shape[1]
        ntasks = len(A)

        if self.sparse:
            Acap = scipy.sparse.csr_matrix((nmodels * ntasks, nmodels * ntasks))
        else:
            Acap = np.zeros((nmodels * ntasks, 1))
        C = np.zeros((nmodels * ntasks, 1))

        beginId = 0
        endId = nmodels
        for task in range(ntasks):
            splI = np.zeros((ntasks, ntasks))
            splI[task][task] = 1  # set this node to 1 so it gets picked up
            # do kronecker product and add to Acap
            newA = np.dot(A[task].T, A[task])
            Acap = Acap + np.kron(splI, newA)
            C[beginId:endId] = np.dot(A[task].T, b[task])
            beginId = endId
            endId = endId + nmodels
        return Acap, C


def shrinkage(a, kappa):
    return np.maximum(0, a - kappa) - np.maximum(0, -a - kappa)


def squaredloss(w, x, y, Omega, lambda_reg):
    '''MSSL with squared loss function '''
    ntasks = Omega.shape[1]
    ndimension = int(len(w) / ntasks)
    wmat = np.reshape(w, (ndimension, ntasks), order='F')

    # make sure data is in the correct format
#    for t in range(ntasks):
#        if len(y[t].shape) > 1:
#            y[t] = np.squeeze(y[t])

    # cost function for each task
    cost = 0
    for t in range(ntasks):
        h_t_x = np.dot(x[t], wmat[:, t])
        cost += ((h_t_x - y[t])**2).mean()
    # cost function regularization
    cost += (0.5 * lambda_reg) * np.trace(np.dot(np.dot(wmat, Omega), wmat.T))
    return cost


def squaredloss_der(w, x, y, Omega, lambda_reg):
    ''' Gradient of the MSSL with squared loss function '''

    ntasks = Omega.shape[1]
    ndimension = int(len(w) / ntasks)
    wmat = np.reshape(w, (ndimension, ntasks), order='F')

    # make sure data is in correct format
#    for t in range(ntasks):
#        if len(y[t].shape) > 1:
#            y[t] = np.squeeze(y[t])

    # gradient of squared loss term
    grad = np.zeros(wmat.shape)
    for t in range(ntasks):
        g1 = np.dot(np.dot(x[t].T, x[t]), wmat[:, t])
        g2 = np.dot(x[t].T, y[t])
        grad[:, t] = (2.0 / x[t].shape[0]) * (g1 - g2)
    # gradient of regularization term
    grad += lambda_reg * np.dot(wmat, Omega)
    grad = np.reshape(grad, (ndimension * ntasks, ), order='F')
    return grad
