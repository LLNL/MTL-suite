
import pandas as pd
import numpy as np
from ..base import BaseEstimator


def check_dataframe(x):
    """ check if input x is a dataframe if so
        convert it to numpy array """
    ntasks = len(x)
    if isinstance(x[0], pd.DataFrame):  # if x is a pandas dataframe
        for t in range(ntasks):
            x[t] = x[t].values.astype(float)
    return x


def normalize_xy_mtl(x, y, d, v, normalize, fit_intercept):

    # number of tasks
    ntasks = len(x)
    x = check_dataframe(x)  # make sure x is not a pandas dataframe

    offsets = {'x_offset': list(),
               'x_scale': list()}

    for t in range(ntasks):
        y[t] = y[t].ravel()
        if d[t] is not None:
            d[t] = d[t].ravel()
            v[t] = v[t].ravel()

    for t in range(ntasks):
        # input (x) variable
        offsets['x_offset'].append(x[t].mean(axis=0))
        if normalize:
            std_x = x[t].std(axis=0)
            std_x[std_x == 0] = 1
            offsets['x_scale'].append(std_x)
        else:
            std = np.ones((x[t].shape[1],))
            offsets['x_scale'].append(std)
        x[t] = (x[t] - offsets['x_offset'][t]) / offsets['x_scale'][t]
        if fit_intercept:
            x[t] = np.hstack((x[t], np.ones((x[t].shape[0], 1))))

    return x, y, d, v, offsets


def normalize_xy_stl(x, y, normalize, fit_intercept):

    # make sure x is not a pandas dataframe
    if isinstance(x, pd.DataFrame):  # if x is a pandas dataframe
        x = x.values.astype(float)

    offsets = {'x_offset': None,
               'x_scale': None}

    # make sure y is a 1-D vector
    y = y.ravel()

    # input (x) variable
    offsets['x_offset'] = x.mean(axis=0)
    if normalize:
        std_x = x.std(axis=0)
        std_x[std_x == 0] = 1
        offsets['x_scale'] = std_x
    else:
        offsets['x_scale'] = 1
    x = (x - offsets['x_offset']) / offsets['x_scale']
    if fit_intercept:
        x = np.hstack((x, np.ones((x.shape[0], 1))))

    return x, y, offsets


class BaseMTLEstimator(BaseEstimator):

    def __init__(self, name, fit_intercept, normalize):
        super().__init__(name, 'MTL')
        self.nb_tasks = -1
        self.nb_dims = -1
        self.normalize = normalize
        self.fit_intercept = fit_intercept

    def fit(self, x, y, **kwargs):
        """ fit model parameters """

        # check if all tasks have the same dimensions
        for k in range(len(x)):
            assert x[k].shape[0] == y[k].shape[0]
            for k2 in range(len(x)):
                msg = ("Dimensions of x[%d] & x[%d] doesnt match: %d - %d" %
                       (k, k2, x[k].shape[1], x[k2].shape[1]))
                assert x[k].shape[1] == x[k2].shape[1], msg

        self.nb_tasks = len(x)  # get number of tasks
        self.nb_dims = x[0].shape[1]  # dimension of the data
        if self.fit_intercept:
            self.nb_dims += 1  # if consider intercept, add another feat +1

        d = kwargs['censor_flag']
        v = kwargs['survival_time']

        x, y, d, v, self.offsets = normalize_xy_mtl(x, y, d, v,
                                                    self.normalize,
                                                    self.fit_intercept)
        self._fit(x, y, censor_flag=d, survival_time=v)  # call child's class specific fit

    def predict(self, x, **kwargs):
        x = check_dataframe(x)
        # make sure x is not a pandas dataframe
        for t in range(self.nb_tasks):
            x[t] = (x[t] - self.offsets['x_offset'][t])
            if self.normalize:
                x[t] = x[t] / self.offsets['x_scale'][t]
            if self.fit_intercept:
                x[t] = np.hstack((x[t], np.ones((x[t].shape[0], 1))))

        yhat = self._predict(x)  # call child's class specific predict

        return yhat


class BaseSingleEstimator(BaseEstimator):

    def __init__(self, name, fit_intercept, normalize):
        super().__init__(name, 'STL')
        self.nb_dims = -1
        self.normalize = normalize
        self.fit_intercept = fit_intercept

    def fit(self, x, y, **kwargs):
        """ fit model parameters """
        self.nb_dims = x.shape[1]  # dimension of the data
        if self.fit_intercept:
            self.nb_dims += 1  # if consider intercept, add another feat +1
        x, y, self.offsets = normalize_xy_stl(x, y, self.normalize,
                                              self.fit_intercept)
        self._fit(x, y, **kwargs)  # call child's class specific fit

    def predict(self, x, **kwargs):
        # make sure x is not a pandas dataframe
        if isinstance(x, pd.DataFrame):
            x = x.values.astype(float)
        x = (x - self.offsets['x_offset'])
        if self.normalize:
            x = x / self.offsets['x_scale']
        if self.fit_intercept:
            x = np.hstack((x, np.ones((x.shape[0], 1))))
        yhat = self._predict(x)  # call child's class specific predict
        return yhat
