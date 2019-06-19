# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-06-17 15:07:03
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-06-18 15:03:15
import sys
sys.path.append('../')
from methods.classifier.mtl.DNNClassifierMTL import DNNClassifierMTL

from methods.classifier.stl.RandomForestClassifier import RandomForestClassifier
from methods.classifier.stl.GradientBoostingClassifier import GradientBoostingClassifier
from methods.classifier.stl.DNNClassifier import DNNClassifier

# from methods.classifier.pooled.PooledLogisticClassifier import PooledLogisticClassifier
from methods.classifier.pooled.PooledRandomForestClassifier import PooledRandomForestClassifier
from methods.classifier.pooled.DNNClassifierPooled import DNNClassifierPooled
from methods.classifier.pooled.PooledLogisticClassifier import PooledLogisticClassifier

from datasets.ArtificialClassificationDatasetMTL import ArtificialClassificationDatasetMTL
from design import ModelTraining


if __name__ == '__main__':

    # set here the list of task-specific datasets
    nb_tasks = 10
    nb_samples = 200
    dimension = 30
    dataset = ArtificialClassificationDatasetMTL(nb_tasks,
                                                 nb_samples,
                                                 dimension)
    dataset.prepare_data()

    # list of methods to compare against
    # note that the same method can be called many times just using
    # different hyper-parameter values
    methods = [
        DNNClassifier(name='DNN-STL'),
        PooledRandomForestClassifier(n_estimators=100, name='Pooled-RF'),
        DNNClassifierPooled(batch_size=32, name='DNN Pooled'),
        DNNClassifierMTL(batch_size=32, name='DNN MTL'),
    ]

    # list of metrics to measure method's performance
    # see list of available metrics in utils/performance_metrics.py
    metrics = ['accuracy']

    exp_folder = __file__.strip('.py')
    exp = ModelTraining(exp_folder)
    exp.execute(dataset, methods, metrics, nb_runs=3)
    exp.generate_report()
