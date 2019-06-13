# -*- coding: utf-8 -*-
# @Author: Andre Ricardo Goncalves
# @Date:   2018-04-24 16:05:40
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-06-13 13:30:45
import sys
sys.path.append('..')
from methods.regressor.mtl.MSSLRegressor import MSSLRegressor

from methods.regressor.stl.LinearRegressor import LinearRegressor
from methods.regressor.stl.GradientBoostingRegressor import GradientBoostingRegressor
from methods.regressor.stl.RandomForestRegressor import RandomForestRegressor

from methods.regressor.pooled.PooledLinearRegressor import PooledLinearRegressor
from methods.regressor.pooled.RandomForestRegressorPooled import RandomForestRegressorPooled

from datasets.ArtificialRegressionDatasetMTL import ArtificialRegressionDatasetMTL
from design import ModelTraining


if __name__ == '__main__':

    # set here the list of task-specific datasets
    nb_tasks = 10
    nb_samples = 1000
    dimension = 30
    dataset = ArtificialRegressionDatasetMTL(nb_tasks,
                                             nb_samples,
                                             dimension)

    dataset.prepare_data()

    # list of methods to compare against
    # note that the same method can be called many times just using
    # different hyper-parameter val ues
    methods = [LinearRegressor(alpha=2e-2, normalize=True,
                               fit_intercept=True, name='LASSO'),
               RandomForestRegressor(n_estimators=100, max_depth=5), ]
    # MSSLRegressor(lambda_1=1e-2, lambda_2=1e-2,
    #               normalize=True, fit_intercept=True,
    #               name='MSSL'),
    # GradientBoostingRegressor(n_estimators=100, max_depth=2,
    #                           normalize=False, fit_intercept=False),
    # RandomForestRegressorPooled(n_estimators=150, max_depth=5),
    # PooledLinearRegressor(alpha_l1=1e-2, normalize=True,
    #                       fit_intercept=True, name='Pooled'), ]

    # list of metrics to measure method's performance
    # see list of available metrics in utils/performance_metrics.py
    metrics = ['rmse']

    exp_folder = __file__.strip('.py')
    exp = ModelTraining(exp_folder)
    exp.execute(dataset, methods, metrics, nb_runs=10)
    exp.generate_report()
