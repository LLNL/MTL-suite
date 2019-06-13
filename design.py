#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:00:05 2018

@author: goncalves1
"""
import os
import types
import pickle
import shutil
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
matplotlib.rcParams.update({'font.size': 11})
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams.update({'figure.autolayout': True})

from abc import ABCMeta, abstractmethod
from UTILS import config, performance_metrics
from UTILS.Logger import Logger
from methods.base import BaseEstimator

# metrics improvement sense
THE_HIGHER_THE_BETTER = {'area_under_curve', 'avg_precision', 'weighted_accuracy', 'accuracy', 'accuracy_per_class', 'c_index', 'c_index_ours'}
THE_LOWER_THE_BETTER = {'rmse', 'nmse', 'rmse_survival', 'mse_survival', 'brier_score', 'mae_survival'} 

# maximum and minimum relative performance improvement
# it is necessary to avoid large differences that distorts the plots
MAX_REL_IMPROVEMENT = 250


class DatasetMTL(object):
    """ """
    __metaclass__ = ABCMeta

    def __init__(self, dataset_name):
        """."""
        self.name = None
        self.data = None
        self.dataset_name = dataset_name

    @abstractmethod
    def prepare_data(self):
        """."""
        pass

    @abstractmethod
    def shuffle_data(self):
        """."""
        pass


class ModelSelection(object):
    """
    Perform model selection by grid-search. Test out a list of hyper-parameter
    values and save the mean performance over all tasks. User must go to
    output folder and check the best hyper-parameter value by looking
    into the resulting .xls file
    """
    def __init__(self, name):

        # make sure all inputs have expected values and types
        assert isinstance(name, str)
        self.name = name
        self.logger = Logger()

    def __check_inputs(self):

        assert isinstance(self.dataset, DatasetMTL)

        # make sure it received a list of methods
        if not isinstance(self.methods, list):
            self.methods = list(self.methods)
        assert len(self.methods) > 0

        # make sure it received a list of metrics
        if not isinstance(self.metrics, list):
            self.metrics = list(self.metrics)
        assert len(self.metrics) > 0

        # check if all methods are valid (instance of Method class)
        for method in self.methods:
            assert isinstance(method, Method)

        # get existing list of available performance metrics
        existing_metrics = [a for a in dir(performance_metrics)
                            if isinstance(performance_metrics.__dict__.get(a),
                                          types.FunctionType)]
        # check if all metrics are valid (exist in performance_metrics module)
        for metric in self.metrics:
            assert metric in existing_metrics

        # number of runs has to be larger then 0
        assert self.nb_runs > 0

    def execute(self, dataset, methods, metrics, nb_runs=1):
        # TODO (goncalves1): run several times <01-05-2017>

        self.dataset = dataset
        self.methods = methods
        self.metrics = metrics
        self.nb_runs = nb_runs

        self.nb_tasks = dataset.get_nb_tasks()

        self.dataset.shuffle_data()

        # set experiment output directory
        directory = os.path.join(config.path_to_output, self.name)
        # if directory already exists, then delete it
        if os.path.exists(directory):
            shutil.rmtree(directory)
        # make a new directory with experiment name
        os.makedirs(directory)

        # experiment log file will be save in 'directory'
        self.logger.set_path(directory)
        self.logger.setup_logger('{}.log'.format(self.name))
        self.logger.info('Experiment directory created.')

        # get list of available metrics
        metric_func = {a: performance_metrics.__dict__.get(a)
                       for a in dir(performance_metrics)
                       if isinstance(performance_metrics.__dict__.get(a),
                                     types.FunctionType)}

        # execute all methods passed through 'methods' attribute
        for method in self.methods:

            self.logger.info('Running {}.'.format(method.name))

            method_str = method.__str__().split('_')[0]
            # set method's output directory
            method_directory = os.path.join(directory, method_str)
            # create directory to save method's results/logs
            os.makedirs(method_directory)

            # inform output directory path to the method
            method.set_output_directory(method_directory)

            for params in method.get_params_grid():
                # set parameter values
                method.set_params(params)
                # method.get_output_directory()

                results = {}
                # initialize metrics dict
                for met in self.metrics:
                    results[met] = list()

                # check model paradigm: STL or MTL
                if method.paradigm == 'STL':
                    # dict to store performance metrics for all tasks
                    # obtained using 'method'
                    for t in range(self.nb_tasks):
                        self.logger.info(('Processing %s' %
                                          self.dataset.data['db_names'][t]))
                        method.fit(self.dataset.data['train']['x'][t],
                                   self.dataset.data['train']['y'][t],
                                   censor_flag=self.dataset.data['train']['censor_flag'][t],
                                   survival_time=self.dataset.data['train']['svv_time'][t],
                                   column_names=self.dataset.column_names,
                                   column_dtypes=self.dataset.column_dtypes)
                        v_t = self.dataset.data['test']['svv_time'][t]
                        d_t = self.dataset.data['test']['censor_flag'][t]
                        y_pred = method.predict(self.dataset.data['test']['x'][t])
                        # dict to save performance metrics for the t-th task
                        for met in self.metrics:
                            y_true = self.dataset.data['test']['y'][t]
                            results[met].append(metric_func[met](y_pred,
                                                                 y_true,
                                                                 censor_flag=d_t,
                                                                 survival_time=v_t))

                    # save result
                elif method.paradigm == 'MTL':
                    method.fit(self.dataset.data['train']['x'].copy(),
                               self.dataset.data['train']['y'].copy(),
                               censor_flag=self.dataset.data['train']['censor_flag'].copy(),
                               survival_time=self.dataset.data['train']['svv_time'].copy(),
                               column_names=self.dataset.column_names,
                               column_dtypes=self.dataset.column_dtypes)
                    y_pred = method.predict(self.dataset.data['test']['x'].copy())
                    for t in range(self.nb_tasks):
                        for met in self.metrics:
                            y_true = self.dataset.data['test']['y'][t]
                            v_t = self.dataset.data['test']['svv_time'][t]
                            d_t = self.dataset.data['test']['censor_flag'][t]
                            results[met].append(metric_func[met](y_pred[t],
                                                                 y_true,
                                                                 censor_flag=d_t,
                                                                 survival_time=v_t))
                else:
                    raise ValueError('Unknown paradigm %s' % (method.paradigm))

                # save results to file
                output_fname = os.path.join(method_directory,
                                            '{}.pkl'.format(method.__str__()))
                with open(output_fname, 'wb') as fh:
                    pickle.dump(results, fh)
                self.logger.info('Results stored in %s' % (output_fname))

    def generate_report(self):
        # read results from experiment folder and store it into a dataframe
        df = self.__read_experiment_results()
        # print(df)

        # save results table into latex format
        txt_filename = os.path.join(config.path_to_output,
                                    self.name,
                                    '{}_table.xls'.format(self.name))
        df.to_excel(txt_filename)

    def __read_experiment_results(self):
        """ Read results from an experiment folder (with multiple methods
        results inside) and place it into a data frame structure.

        Args:
            experiment(str): name of the experiment in 'outputs' directory
        """
        experiment_dir = os.path.join(config.path_to_output, self.name)

        # list that will contain all results information as a table
        # this list will be inserted in a pandas dataframe to become
        # easier to generate plots and latex tables
        result_contents = list()

        # iterate over the methods
        # method definition here is "an execution" of a method
        # the same method (let's say Linear Regression) has two instances
        # with different hyper-parameter values, then there will be two
        # 'methods' here (or two entries in the results table)
        for method in os.listdir(experiment_dir):
            method_dir = os.path.join(experiment_dir, method)
            if os.path.isdir(method_dir):
                # get results filename (the one ending in 'pkl')
                for resf in os.listdir(method_dir):
                    if resf.endswith('.pkl'):
                        method_str, hyp_pars = resf[:-4].split('_', maxsplit=1)
                        with open(os.path.join(method_dir, resf), 'rb') as fh:
                            results = pickle.load(fh)
                            # iterate over metrics for the given method
                            for k in results.keys():
                                np_res = np.array(results[k])
                                result_contents.append([method_str, hyp_pars,
                                                        k, np_res.mean(),
                                                        np_res.std(),
                                                        np_res.min(),
                                                        np_res.max()])
        # store result_contents list into a dataframe for easier manipulation
        column_names = ['Method', 'Params', 'Metric', 'Mean',
                        'StD', 'Min', 'Max']
        return pd.DataFrame(result_contents, columns=column_names)


class ModelTraining(object):
    """
    """
    def __init__(self, name):

        assert isinstance(name, str)

        self.name = name
        self.dataset = None
        self.methods = None
        self.metrics = None
        self.nb_runs = -1
        self.nb_tasks = -1

        self.logger = Logger()

    def execute(self, dataset, methods, metrics, nb_runs=1, only_report=False):

        self.__check_inputs(dataset, methods, metrics, nb_runs)
        self.dataset = dataset
        self.methods = methods
        self.metrics = metrics
        self.nb_tasks = dataset.get_nb_tasks()

        self.nb_runs = nb_runs

        if only_report:
            return 0
        # set experiment output directory
        directory = os.path.join(config.path_to_output, self.name)
        # if directory already exists, then delete it
        if os.path.exists(directory):
            shutil.rmtree(directory)
        # make a new directory with experiment name
        os.makedirs(directory)

        # experiment log file will be save in 'directory'
        self.logger.set_path(directory)
        self.logger.setup_logger('{}.log'.format(self.name))
        self.logger.info('Experiment directory created.')

        # get list of available metrics
        metric_func = {a: performance_metrics.__dict__.get(a)
                       for a in dir(performance_metrics)
                       if isinstance(performance_metrics.__dict__.get(a),
                                     types.FunctionType)}

        for r_i in range(self.nb_runs):

            self.logger.info('Executing \'Run {}\'.'.format(r_i+1))

            # shuffle and re-split the data between training and test
            self.dataset.shuffle_data()

            run_directory = os.path.join(directory, 'run_{}'.format(r_i+1))

            # execute all methods passed through 'methods' attribute
            for method in self.methods:

                self.logger.info('Running {}.'.format(method.name))

                # set method's output directory
                method_directory = os.path.join(run_directory, method.__str__())
                # create directory to save method's results/logs
                os.makedirs(method_directory)

                # inform output directory path to the method
                method.set_output_directory(method_directory)

                # check model paradigm: STL or MTL
                if method.paradigm == 'STL':
                    # dict to store performance metrics for all tasks
                    # obtained using 'method'
                    results = {}
                    ypred_yobs = {}
                    for t in range(self.nb_tasks):
                        self.logger.info(('Processing %s' %
                                          self.dataset.data['db_names'][t]))
                        # print('[Y] Shape: {} \t Sum: {}'.format(self.dataset.data['train']['y'][t].shape,
                                                                # self.dataset.data['train']['y'][t].sum()))
                        # print(self.dataset.data['train']['x'][t].columns)
                        method.fit(self.dataset.data['train']['x'][t],
                                   self.dataset.data['train']['y'][t],
                                   censor_flag=self.dataset.data['train']['censor_flag'][t],
                                   survival_time=self.dataset.data['train']['svv_time'][t],
                                   column_names=self.dataset.column_names,
                                   column_dtypes=self.dataset.column_dtypes,
                                   unique_categ_values=self.dataset.unique_values_categ,
                                   run=r_i, task_id=t)

                        y_pred = method.predict(self.dataset.data['test']['x'][t], task_id=t)
                        v_t = self.dataset.data['test']['svv_time'][t]
                        d_t = self.dataset.data['test']['censor_flag'][t]
                        sample_id = self.dataset.data['test']['sample_id'][t]
                        ypred_yobs[self.dataset.data['db_names'][t]] = {'pred': y_pred,
                                                                        'obs': self.dataset.data['test']['y'][t],
                                                                        'svv_time': v_t,
                                                                        'censor_flag': d_t,
                                                                        'sample_id': sample_id}

                        # dict to save performance metrics for the t-th task
                        result_task = {}
                        for met in self.metrics:
                            y_true = self.dataset.data['test']['y'][t]
                            result_task[met] = metric_func[met](y_pred,
                                                                y_true,
                                                                censor_flag=self.dataset.data['test']['censor_flag'][t],
                                                                survival_time=self.dataset.data['test']['svv_time'][t])
                        n_samples = self.dataset.data['train']['x'][t].shape[0]
                        results[self.dataset.data['db_names'][t]] = {'results': result_task,
                                                                     'sample_size': n_samples}
                    # save result
                elif method.paradigm == 'MTL':
                    method.fit(self.dataset.data['train']['x'].copy(),
                               self.dataset.data['train']['y'].copy(),
                               censor_flag=self.dataset.data['train']['censor_flag'].copy(),
                               survival_time=self.dataset.data['train']['svv_time'].copy(),
                               column_names=self.dataset.column_names,
                               column_dtypes=self.dataset.column_dtypes,
                               unique_categ_values=self.dataset.unique_values_categ,
                               run=r_i)
                    y_pred = method.predict(self.dataset.data['test']['x'].copy())
                    results = {}
                    ypred_yobs = {}
                    for t in range(self.nb_tasks):
                        result_task = {}
                        y_true = self.dataset.data['test']['y'][t].copy()
                        v_t = self.dataset.data['test']['svv_time'][t]
                        d_t = self.dataset.data['test']['censor_flag'][t]
                        sample_id = self.dataset.data['test']['sample_id'][t]
                        ypred_yobs[self.dataset.data['db_names'][t]] = {'pred': y_pred[t],
                                                                        'obs': y_true,
                                                                        'svv_time': v_t,
                                                                        'censor_flag': d_t,
                                                                        'sample_id': sample_id}
                        for met in self.metrics:
                            result_task[met] = metric_func[met](y_pred[t],
                                                                y_true,
                                                                censor_flag=self.dataset.data['test']['censor_flag'][t],
                                                                survival_time=self.dataset.data['test']['svv_time'][t])
                        n_samples = self.dataset.data['train']['x'][t].shape[0]
                        results[self.dataset.data['db_names'][t]] = {'results': result_task,
                                                                     'sample_size': n_samples}

                else:
                    raise ValueError('Unknown paradigm %s' % (method.paradigm))

                # save predicted and observed values to file
                output_fname1 = os.path.join(method_directory,
                                             '{}_pred_obs.dat'.format(method.__str__()))
                with open(output_fname1, 'wb') as fh:
                    pickle.dump(ypred_yobs, fh)

                # save all performances to file
                output_fname = os.path.join(method_directory,
                                            '{}.pkl'.format(method.__str__()))
                with open(output_fname, 'wb') as fh:
                    pickle.dump(results, fh)
                self.logger.info('Results stored in %s' % (output_fname))

    def generate_report(self):
        # read results from experiment folder and store it into a dataframe
        df, op_dict = self.__read_experiment_results()

        # save results table into latex format
        txt_filename = os.path.join(config.path_to_output,
                                    self.name,
                                    '{}_table.tex'.format(self.name))
        with open(txt_filename, 'w') as fh:
            fh.write(df.to_latex())

        # set output pdf name
        pdf_filename = os.path.join(config.path_to_output,
                                    self.name,
                                    '{}_report.pdf'.format(self.name))

        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)

        # call several plot functions
        # self.__tasks_average_std_plot(df, pdf)
        self.__performance_plots_per_method(df, op_dict, pdf)
        # self.__pooled_performance_plots(df, op_dict, pdf)
        self.__performance_plots_per_task(df, op_dict, pdf)
        # self.__individual_tasks_plot(df, op_dict, pdf)
        # self.__methods_scatter_plot(df, pdf)
        # self.__methods_diff_bars_plot(df, pdf)
        # self.__plot_precision_recall_curve(df, op_dict, pdf)

        # close pdf file
        pdf.close()

    def __check_inputs(self, dataset, methods, metrics, nb_runs):
        # make sure all inputs have expected values and types
        assert isinstance(dataset, DatasetMTL)

        # make sure it received a list of methods
        if not isinstance(methods, list):
            methods = list(methods)
        assert len(methods) > 0

        # make sure it received a list of metrics
        if not isinstance(metrics, list):
            metrics = list(metrics)
        assert len(metrics) > 0

        # check if all methods are valid (instance of Method class)
        for method in methods:
            assert isinstance(method, BaseEstimator)

        # get existing list of available performance metrics
        existing_metrics = [a for a in dir(performance_metrics)
                            if isinstance(performance_metrics.__dict__.get(a),
                                          types.FunctionType)]
        # check if all metrics are valid (exist in performance_metrics module)
        for metric in metrics:
            assert metric in existing_metrics

        # number of runs has to be larger then 0
        assert nb_runs > 0


    def __read_experiment_results(self):
        """ Read results from an experiment folder (with multiple methods
        results inside) and place it into a data frame structure.

        Args:
            experiment(str): name of the experiment in 'outputs' directory
        """
        experiment_dir = os.path.join(config.path_to_output, self.name)

        # list that will contain all results information as a table
        # this list will be inserted in a pandas dataframe to become
        # easier to generate plots and latex tables
        result_contents = list()
        obs_pred = dict()

        for run in next(os.walk(experiment_dir))[1]:
            obs_pred[run] = dict()
            run_dir = os.path.join(experiment_dir, run)
            # iterate over the methods
            # method definition here is "an execution" of a method
            # the same method (let's say Linear Regression) has two instances
            # with different hyper-parameter values, then there will be two
            # 'methods' here (or two entries in the results table)
            for method in next(os.walk(run_dir))[1]:
                method_dir = os.path.join(run_dir, method)
                # get results filename (the one ending with 'pkl')
                resf = [f for f in os.listdir(method_dir) if f.endswith('.pkl')][0]
                with open(os.path.join(method_dir, resf), 'rb') as fh:
                    # dict with each task result as a key
                    # for each key is assigned a dict w/ task specific results
                    tasks_results = pickle.load(fh)
                    for k in tasks_results.keys():
                        # iterate over metrics for k-th task
                        for m in tasks_results[k]['results'].keys():
                            result_contents.append([run, method, method, k, m,
                                                    tasks_results[k]['results'][m],
                                                    tasks_results[k]['sample_size']])
                # get results filename (the one ending in 'dat')
                resf = [f for f in os.listdir(method_dir) if f.endswith('.dat')][0]

                with open(os.path.join(method_dir, resf), 'rb') as fh:
                    tasks_results = pickle.load(fh)
                    obs_pred[run][method] = tasks_results

        # store result_contents list into a dataframe for easier manipulation
        column_names = ['Run', 'Method', 'Methods', 'Task', 'Metric', 'Value', 'SampleSize']
        df = pd.DataFrame(result_contents, columns=column_names)
        return df, obs_pred


    def __tasks_average_std_plot(self, df, pdf):
        """ Plot tasks average and std results.

        It plots a single plot with the average and std results over all tasks.

        Args:
            df (pandas.DataFrame): panda's dataframe containing the results.
            pdf (matplotlib.backend.pdf): matplotlib object to write plots into
                    a pdf file.
            title (string): plot title.
        """
        df.is_copy = None
        runs = df['Run'].unique()
        methods = df['Method'].unique()
        methods_abrrev = df['Methods'].unique()
        tasks = df['Task'].unique()
        metrics = self.metrics

        metric_map = {'area_under_curve_uncensored': 'AUC',
                      'avg_precision_uncensored': 'AP',
                      'brier_score': 'Brier score'}

        for metric in metrics:
            df_m = df.loc[df["Metric"] == metric]
            perf_mat = np.zeros((len(runs), len(methods), len(tasks)))
            for i, r in enumerate(runs):
                for j, m in enumerate(methods):
                    df_ij = df_m.loc[(df_m['Run'] == r) & (df_m['Method'] == m)]
                    perf_mat[i, j, :] = df_ij['Value'].to_numpy()

            colors = ['azure', 'green', 'sienna', 'orchid', 'darkblue']
            colors += ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
            fig, ax1 = plt.subplots()
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            bplot = plt.boxplot(perf_mat.mean(axis=2), patch_artist=True)

            plt.xticks(1+np.arange(len(methods_abrrev)),
                       methods_abrrev, fontsize=15)
            plt.ylabel(metric_map[metric], fontsize=15)

            for i, (p1, p2) in enumerate(zip(bplot['boxes'], bplot['fliers'])):
                p1.set(facecolor='xkcd:{}'.format(colors[i]))
                p2.set(color='xkcd:{}'.format(colors[i]))
            pdf.savefig(fig)


    def __methods_scatter_plot(self, df, pdf):
        metrics = self.metrics
        for metric in metrics:
            df_s = df.loc[df["Metric"] == metric]
            methods = df_s['Method'].unique()
            for i, met1 in enumerate(methods):
                for j, met2 in enumerate(methods):
                    if i < j:
                        df_s1 = df_s.loc[df["Method"] == met1]
                        df_s2 = df_s.loc[df["Method"] == met2]
                        runs = df_s1['Run'].unique()
                        runs_m1 = np.zeros((self.nb_tasks, len(runs)))
                        runs_m2 = np.zeros((self.nb_tasks, len(runs)))
                        for k,run in enumerate(runs):
                            runs_m1[:, k] = df_s1.loc[df_s1['Run'] == run, 'Value'].to_numpy()
                            runs_m2[:, k] = df_s2.loc[df_s2['Run'] == run, 'Value'].to_numpy()
                        mean_m1 = runs_m1.mean(axis=1)
                        mean_m2 = runs_m2.mean(axis=1)
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        ax.plot(mean_m1, mean_m2, 'bo')
                        xmin = min(np.min(mean_m1), np.min(mean_m2))
                        xmax = max(np.max(mean_m1), np.max(mean_m2))
                        ax.plot([xmin*0.8, xmax*1.2],
                                [xmin*0.8, xmax*1.2],
                                ls="--", c=".3")
                        tasks_name = df_s1['Task'].unique()
                        for l, txt in enumerate(tasks_name):
                            ax.annotate(txt[4:], (mean_m1[l],
                                                  mean_m2[l]),
                                        fontsize=8)
                        # ax.set_title('%s: Method X Method' % (metric))
                        ax.set_xlabel(met1.split('_')[0])
                        ax.set_ylabel(met2.split('_')[0])
                        pdf.savefig(fig)


    def __methods_diff_bars_plot(self, df, pdf):
        fig = plt.figure()
        metrics = self.metrics
        for metric in metrics:
            df_s = df.loc[df["Metric"] == metric]
            methods = df_s['Method'].unique()
            for i, met1 in enumerate(methods):
                for j, met2 in enumerate(methods):
                    if i != j:
                        df_s1 = df_s.loc[df["Method"] == met1]
                        df_s2 = df_s.loc[df["Method"] == met2]

                        runs = df_s1['Run'].unique()
                        runs_m1 = np.zeros((self.nb_tasks, len(runs)))
                        runs_m2 = np.zeros((self.nb_tasks, len(runs)))
                        for k, run in enumerate(runs):
                            if k == 0:
                                ssize = df_s2.loc[df_s2['Run'] == run, 'SampleSize'].to_numpy()
                            runs_m1[:, k] = df_s1.loc[df_s1['Run'] == run, 'Value'].to_numpy()
                            runs_m2[:, k] = df_s2.loc[df_s2['Run'] == run, 'Value'].to_numpy()
                        s_ids = np.argsort(ssize)

                        ax = fig.add_subplot(1, 1, 1)
                        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
                        # relative performance
                        delta = np.divide((runs_m1 - runs_m2), runs_m1, out=np.zeros_like(runs_m1), where=runs_m1 != 0) * 100
                        delta = np.minimum(np.maximum(delta, -MAX_REL_IMPROVEMENT), MAX_REL_IMPROVEMENT)
                        delta_mean = delta.mean(axis=1)
                        delta_mean = delta_mean.T[s_ids].T  # reorder columns
                        if metric in THE_HIGHER_THE_BETTER:
                            colors = ['g' if d >= 0 else 'r' for d in delta_mean]
                        else:
                            colors = ['g' if d <= 0 else 'r' for d in delta_mean]

                        barp = plt.bar(np.arange(len(delta_mean)),
                                       delta_mean, color=colors,
                                       yerr=delta.std(axis=1),
                                       error_kw=dict(lw=1))

                        # Add counts above the two bar graphs
                        for k, rect in enumerate(barp):
                            plt.text(rect.get_x() + rect.get_width()/2.0, 
                                     0, '%d' % ssize[s_ids[k]],
                                     ha='center', va='bottom', fontsize=10,
                                     color='k')

                        xticks = [xt for xt in df_s1['Task'].unique().tolist()]
                        # xticks = [t.replace('_', '\n') for t in xticks]

                        xticks = np.array(xticks)
                        xi = np.arange(delta.shape[0])
                        plt.xticks(xi, xticks[s_ids])
                        locs, labels = plt.xticks()
                        plt.setp(labels, rotation=90)
                        ax.tick_params(axis='x', which='major', labelsize=6)

                        title_txt = '{} vs {}'.format(met1.split('_')[0],
                                                      met2.split('_')[0])
                        ax.set_title(title_txt, fontsize=20)
                        metric_name = ' '.join(metric.split('_')).title()
                        ax.set_ylabel('{} \n Relative performance (%)'.format(metric_name), fontsize=15)

                        #plt.tight_layout()
                        pdf.savefig(fig)
                        plt.clf()


    def __pooled_performance_plots(self, df, op_dict, pdf):
        """ Compute performance metric of all test samples regardless the task.
        Data from all tasks are pooled and then the performance is computed. """
        # get list of available metrics
        metric_func = {a: performance_metrics.__dict__.get(a)
                       for a in dir(performance_metrics)
                       if isinstance(performance_metrics.__dict__.get(a),
                                     types.FunctionType)}

        tasks = df['Task'].unique()
        runs = op_dict.keys()
        methods = op_dict[list(runs)[0]].keys()

        for i, metric in enumerate(self.metrics):
            perform = list()
            for r, run in enumerate(runs):
                for j, method in enumerate(methods):
                    # accumulate performance from all tasks
                    accum_pred = np.array([])
                    accum_observed = np.array([])
                    accum_v = np.array([])
                    accum_d = np.array([])
                    for task in tasks:
                        accum_pred = np.concatenate((accum_pred, op_dict[run][method][task]['pred'].ravel()))
                        accum_observed = np.concatenate((accum_observed, op_dict[run][method][task]['obs'].ravel()))
                        accum_v = np.concatenate((accum_v, op_dict[run][method][task]['svv_time'].ravel()))
                        accum_d = np.concatenate((accum_d, op_dict[run][method][task]['censor_flag'].ravel()))
                    # compute performance from data from all tasks
                    v = metric_func[metric](accum_pred,
                                            accum_observed,
                                            censor_flag=accum_d,
                                            survival_time=accum_v)
                    perform.append([metric, run, method, v])

            column_names = ['Metric', 'Run', 'Method', 'Value']
            df_perf = pd.DataFrame(perform, columns=column_names)
            with sns.plotting_context("notebook", font_scale=1.3):
                g = sns.catplot(x="Method", y="Value", data=df_perf,
                                kind="bar", palette="colorblind")
                g.set_axis_labels("", ' '.join(metric.split('_')).title())  # .set(ylim=(0.7, 0.8))
                plt.tight_layout()
            pdf.savefig(g.fig)
            plt.clf()


    def __performance_plots_per_method(self, df, op_dict, pdf):
        """ Compute performance metric of all test samples regardless the task.
        Data from all tasks are pooled and then the performance is computed. """

        # get list of available metrics
        metric_func = {a: performance_metrics.__dict__.get(a)
                       for a in dir(performance_metrics)
                       if isinstance(performance_metrics.__dict__.get(a),
                                     types.FunctionType)}

        tasks = df['Task'].unique()
        runs = op_dict.keys()
        methods = op_dict[list(runs)[0]].keys()

        metric_map = {'area_under_curve_uncensored': 'AUC',
                      'avg_precision_uncensored': 'AP',
                      'brier_score': 'Brier score'}

        # qualitative_colors = sns.color_palette("Set3", 10)
        for i, metric in enumerate(self.metrics):
            perform = list()
            for r, run in enumerate(runs):
                for j, method in enumerate(methods):
                    v = 0
                    for task in tasks:
                        # compute performance from data from all tasks
                        v += metric_func[metric](op_dict[run][method][task]['pred'],
                                                 op_dict[run][method][task]['obs'],
                                                 censor_flag=op_dict[run][method][task]['censor_flag'],
                                                 survival_time=op_dict[run][method][task]['svv_time'])
                    perform.append([metric, task, run, method, v / len(tasks)])

            column_names = ['Metric', 'Task', 'Run', 'Method', 'Value']
            df_perf = pd.DataFrame(perform, columns=column_names)
            flierprops = dict(markerfacecolor='0.75', markersize=2, linestyle='none')
            with sns.plotting_context("notebook", font_scale=1.3):
                g = sns.catplot(x="Method", y="Value", data=df_perf,
                                kind="box", palette='colorblind', linewidth=0.01,
                                width=0.4, legend=False, flierprops=flierprops)
                # g.despine(left=True)
                # g.despine(left=True, bottom=True)
                # g.set_xticklabels(rotation=90)
                if metric in metric_map.keys():
                    g.set_axis_labels("", metric_map[metric])
                else:
                    g.set_axis_labels("", metric)
            g.fig.set_size_inches(8, 4)
            if metric in THE_HIGHER_THE_BETTER:
                plt.ylim([0, 1])

            plt.tight_layout()
            pdf.savefig(g.fig)
            plt.clf()

    def __performance_plots_per_task(self, df, op_dict, pdf):
        """ Compute performance metric of all test samples regardless the task.
        Data from all tasks are pooled and then the performance is computed. """

        # get list of available metrics
        metric_func = {a: performance_metrics.__dict__.get(a)
                       for a in dir(performance_metrics)
                       if isinstance(performance_metrics.__dict__.get(a),
                                     types.FunctionType)}

        # tasks = df['Task'].unique()
        tasks_orig = df['Task'].unique()
        tasks_new = ['T{}'.format(i+1) for i in range(len(df['Task'].unique()))]
        tasks_map = dict(zip(tasks_orig, tasks_new))

        runs = op_dict.keys()
        methods = op_dict[list(runs)[0]].keys()

        metric_map = {'area_under_curve_uncensored': 'AUC',
                      'avg_precision_uncensored': 'AP',
                      'brier_score': 'Brier score'}

        # qualitative_colors = sns.color_palette("Set3", 10)
        for i, metric in enumerate(self.metrics):
            perform = list()
            for r, run in enumerate(runs):
                for j, method in enumerate(methods):
                    for task in tasks_orig:
                        # compute performance from data from all tasks
                        v = metric_func[metric](op_dict[run][method][task]['pred'],
                                                op_dict[run][method][task]['obs'],
                                                censor_flag=op_dict[run][method][task]['censor_flag'],
                                                survival_time=op_dict[run][method][task]['svv_time'])
                        perform.append([metric, tasks_map[task], run, method, v])

            column_names = ['Metric', 'Task', 'Run', 'Method', 'Value']
            df_perf = pd.DataFrame(perform, columns=column_names)
            flierprops = dict(markerfacecolor='0.75', markersize=2, linestyle='none')
            with sns.plotting_context("notebook", font_scale=1.3):
                g = sns.catplot(x="Task", y="Value", hue='Method', data=df_perf,
                                kind="box", palette='colorblind', linewidth=0.01,
                                width=0.7, legend=False, flierprops=flierprops)
                # g.despine(left=True)
                # g.set_xticklabels(rotation=90)
                if metric in metric_map.keys():
                    g.set_axis_labels("", metric_map[metric])
                else:
                    g.set_axis_labels("", metric)
            g.fig.set_size_inches(8, 4)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.tight_layout()
            pdf.savefig(g.fig)
            plt.clf()

    def __individual_tasks_plot(self, df, op_dict, pdf):
        """ Plot results for each task separately.

        It plots results for all metrics for each task separately.
        Each task will have its own plot and page in the pdf report file.

        Args:
            df (pandas.DataFrame): results data frame
            pdf (matplotlib.backend.pdf): matplotlib object to write plots into
                    a pdf file.
        """
        def __kpmeier(x):
            x_uniq = np.unique(x)
            x0 = list([x_uniq[0]])
            x1 = list([1])
            for t in x_uniq[:-1]:
                s_hat_ti = list()
                for ti in x_uniq:
                    if ti < t:
                        d_i = (x == ti).sum()
                        n_i = (x > ti).sum()
                        s_hat_ti.append((1 - d_i/float(n_i)))
                x0.extend((t, t))
                x1.extend((x1[-1], np.array(s_hat_ti).prod()))
            x0.extend((x0[-1], x_uniq[-1], ))
            x1.extend((0, 0))
            return x0, x1

        def __individual_metric_kaplan_meier(df, op_task, pdf, title):
            df.is_copy = None
            # fig, ax = plt.subplots(figsize=(10, 5), ncols=2, nrows=1)
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            ax1 = fig.add_subplot(2, 1, 1)
            plt.suptitle(title, fontsize=16)

            # Draw a nested barplot to show Class. ACC for each methods
            g = sns.factorplot(x="Metric", y="Value", hue="Methods",
                               data=df, size=6, kind="bar",
                               palette="muted", legend=False, ax=ax1)
            g.despine(left=True)
            g.set_ylabels('Value')
            plt.legend(loc='upper right')
            if not title:
                nb_tasks = len(df['Task'].unique())
                g.fig.suptitle('Average over all ({}) tasks'.format(nb_tasks))
            else:
                g.fig.suptitle(title)

            # kaplan meier
            if df['Metric'].iloc[0] in ('rmse', 'nmse'):
                # colors = ['azure', 'green', 'sienna', 'orchid', 'darkblue']
                colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
                leg_list = list()
                ax2 = fig.add_subplot(2, 1, 2)
                for i, k in enumerate(op_task.keys()):
                    # plot observed curve
                    if i == 0:
                        y0, y1 = __kpmeier(op_task[k]['obs'])
                        ax2.plot(y0, y1, color='xkcd:black')
                        leg_list.append('Observed')

                    # plot predicted value for all methods
                    p0, p1 = __kpmeier(np.squeeze(op_task[k]['pred']))
                    ax2.plot(p0, p1, color='xkcd:%s' % colors[i])
                    leg_list.append(k[0:10])
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Survival Probability (%)')
                ax2.legend(leg_list)
            pdf.savefig(fig)

        df.is_copy = None
        tasks = df['Task'].unique()
        runs = op_dict.keys()
        for task in tasks:
            op_dict_task = dict()
            for met in op_dict.keys():
                print(op_dict[met].keys())
                op_dict_task[met] = op_dict[met][task]
            __individual_metric_kaplan_meier(df[df['Task'] == task],
                                             op_dict_task, pdf, task)

    def __plot_prediction_distribution(self, df, op_dict, pdf):
        """ Compute performance metric of all test samples regardless the task.
        Data from all tasks are pooled and then the performance is computed. """
        # get list of available metrics
        # metric_func = {a: performance_metrics.__dict__.get(a)
        #                for a in dir(performance_metrics)
        #                if isinstance(performance_metrics.__dict__.get(a),
        #                              types.FunctionType)}

        tasks = df['Task'].unique()
        runs = op_dict.keys()
        methods = op_dict[list(runs)[0]].keys()

        colors = ["b", "r", "y", "g", "m", "k"]
        fig = plt.figure()
        precision = {}
        recall = {}
        for j, method in enumerate(methods):
            for r, run in enumerate(runs):
                # accumulate performance from all tasks
                accum_pred = np.array([])
                accum_observed = np.array([])
                accum_v = np.array([])
                accum_d = np.array([])
                for task in tasks:
                    accum_pred = np.concatenate((accum_pred, op_dict[run][method][task]['pred'].ravel()))
                    accum_observed = np.concatenate((accum_observed, op_dict[run][method][task]['obs'].ravel()))
                    accum_v = np.concatenate((accum_v, op_dict[run][method][task]['svv_time'].ravel()))
                    accum_d = np.concatenate((accum_d, op_dict[run][method][task]['censor_flag'].ravel()))
            precision[method], recall[method], _ = precision_recall_curve(accum_observed, accum_pred)

            plt.step(recall[method], precision[method], color=colors[j], where='post', label=method)
            # plt.fill_between(recall[method], precision[method], step='post', alpha=0.2, color='b')

        

            #         # compute performance from data from all tasks
            #         v = metric_func[metric](accum_pred, 
            #                                 accum_observed,
            #                                 censor_flag=accum_d,
            #                                 survival_time=accum_v)
            #         perform.append([metric, run, method, v])

            # column_names = ['Metric', 'Run', 'Method', 'Value']
            # df_perf = pd.DataFrame(perform, columns=column_names)
            # with sns.plotting_context("notebook", font_scale=1.3):
            #     g = sns.catplot(x="Method", y="Value", data=df_perf,
            #                     kind="bar", palette=sns.xkcd_palette(colors))
            #     g.set_axis_labels("", ' '.join(metric.split('_')).title()) #.set(ylim=(0.7, 0.8))
            #     plt.tight_layout()

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend()
        # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        pdf.savefig(fig)
        plt.clf()


    def __plot_precision_recall_curve(self, df, op_dict, pdf):
        """ Compute performance metric of all test samples regardless the task.
        Data from all tasks are pooled and then the performance is computed. """
        # get list of available metrics
        # metric_func = {a: performance_metrics.__dict__.get(a)
        #                for a in dir(performance_metrics)
        #                if isinstance(performance_metrics.__dict__.get(a),
        #                              types.FunctionType)}

        tasks = df['Task'].unique()
        runs = op_dict.keys()
        methods = op_dict[list(runs)[0]].keys()

        colors = ["b", "r", "y", "g", "m", "k"]
        fig = plt.figure()
        precision = {}
        recall = {}
        for j, method in enumerate(methods):
            for r, run in enumerate(runs):
                # accumulate performance from all tasks
                accum_pred = np.array([])
                accum_observed = np.array([])
                accum_v = np.array([])
                accum_d = np.array([])
                for task in tasks:
                    accum_pred = np.concatenate((accum_pred, op_dict[run][method][task]['pred'].ravel()))
                    accum_observed = np.concatenate((accum_observed, op_dict[run][method][task]['obs'].ravel()))
                    accum_v = np.concatenate((accum_v, op_dict[run][method][task]['svv_time'].ravel()))
                    accum_d = np.concatenate((accum_d, op_dict[run][method][task]['censor_flag'].ravel()))
            precision[method], recall[method], _ = precision_recall_curve(accum_observed, accum_pred)

            plt.step(recall[method], precision[method], color=colors[j], where='post', label=method)
            # plt.fill_between(recall[method], precision[method], step='post', alpha=0.2, color='b')

        

            #         # compute performance from data from all tasks
            #         v = metric_func[metric](accum_pred, 
            #                                 accum_observed,
            #                                 censor_flag=accum_d,
            #                                 survival_time=accum_v)
            #         perform.append([metric, run, method, v])

            # column_names = ['Metric', 'Run', 'Method', 'Value']
            # df_perf = pd.DataFrame(perform, columns=column_names)
            # with sns.plotting_context("notebook", font_scale=1.3):
            #     g = sns.catplot(x="Method", y="Value", data=df_perf,
            #                     kind="bar", palette=sns.xkcd_palette(colors))
            #     g.set_axis_labels("", ' '.join(metric.split('_')).title()) #.set(ylim=(0.7, 0.8))
            #     plt.tight_layout()


        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend()
        # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        pdf.savefig(fig)
        plt.clf()


#    def __pooled_performance_plots(self, df, op_dict, pdf):
#        # get list of available metrics
#        metric_func = {a: performance_metrics.__dict__.get(a)
#                       for a in dir(performance_metrics)
#                       if isinstance(performance_metrics.__dict__.get(a),
#                                     types.FunctionType)}
#
#        tasks = df['Task'].unique()
#        accum_samples_perf = dict()
#        runs = op_dict.keys()
#        methods = op_dict[list(runs)[0]].keys()
#        for r, run in enumerate(runs):  # iterate over runs
#            accum_samples_perf[run] = dict()
#            for i, met in enumerate(op_dict[run].keys()):
#                accum_method = list()
#                accum_observed = list()
#                accum_d = list()
#                accum_v = list()
#                for task in tasks:
#                    accum_method.append(op_dict[run][met][task]['pred'])
#                    if i == 0:
#                        accum_observed.append(op_dict[run][met][task]['obs'])
#                        accum_v.append(op_dict[run][met][task]['svv_time'])
#                        accum_d.append(op_dict[run][met][task]['censor_flag'])
#                if i == 0:
#                    accum_samples_perf[run]['obs'] = np.hstack(accum_observed)
#                    accum_samples_perf[run]['svv_time'] = np.hstack(accum_v)
#                    accum_samples_perf[run]['censor_flag'] = np.hstack(accum_d)
#                accum_samples_perf[run][met] = np.hstack(accum_method)
#
#        fig = plt.figure()
#        fig.subplots_adjust(hspace=0.4, wspace=0.4)
#        width = 0.75
#        axs = [None]*len(self.metrics)
#        for i, metr in enumerate(self.metrics):
#            methods_perf = np.zeros((len(methods), len(runs)))
#            for r, run in enumerate(runs):
#                for j, method in enumerate(methods):
#                    v = metric_func[metr](accum_samples_perf[run][method],
#                                          accum_samples_perf[run]['obs'],
#                                          censor_flag=accum_samples_perf[run]['censor_flag'],
#                                          survival_time=accum_samples_perf[run]['svv_time'])
#                    methods_perf[j, r] = v
#
#            axs[i] = fig.add_subplot(len(self.metrics), 1, i+1)
#            index = np.arange(len(methods))
#            offset = 0.35
#            axs[i].barh(index+offset, methods_perf.mean(axis=1), height=0.5,
#                        xerr=methods_perf.std(axis=1), align='center')
#
#            axs[i].set_xlim([0, 1])
#            axs[i].set_yticks(index+width/2)
#            axs[i].set_yticklabels([m.split('_')[0] for m in methods],
#                                   minor=False)
#
#            for k, v in enumerate(methods_perf.mean(axis=1)):
#                axs[i].text(v + 0.1, k + 0.3, '%.2f' % v,
#                            color='red',
#                            fontweight='bold')
#            metric_name = ' '.join(metr.split('_')).title()
#            axs[i].set_title('{} on all samples'.format(metric_name))
##            axs[i].set_ylabel('{}'.format(metr))
#        plt.tight_layout()
#        pdf.savefig(fig)
