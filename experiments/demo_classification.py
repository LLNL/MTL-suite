# -*- coding: utf-8 -*-
# @Author: Andre Ricardo Goncalves
# @Date:   2018-10-22 13:29:43
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-06-17 15:58:21
import sys
sys.path.append('../')
from methods.classifier.mtl.MSSLClassifier import MSSLClassifier
from methods.classifier.mtl.JFSMTLClassifier import JFSMTLClassifier

from methods.classifier.stl.LogisticClassifier import LogisticClassifier
from methods.classifier.stl.RandomForestClassifier import RandomForestClassifier
from methods.classifier.stl.GradientBoostingClassifier import GradientBoostingClassifier

from methods.classifier.pooled.PooledLogisticClassifier import PooledLogisticClassifier
from methods.classifier.pooled.PooledRandomForestClassifier import PooledRandomForestClassifier

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
  methods = [PooledLogisticClassifier(alpha_l1=0.05, name='Pooled-LogReg'),
             MSSLClassifier(lambda_1=0.01, lambda_2=0.01, name='MSSL'),
             LogisticClassifier(alpha_l1=0.02, name='LogReg-STL'),
             JFSMTLClassifier(rho_L21=0.1, rho_L2=0, name='JFSMTL'),
             PooledRandomForestClassifier(n_estimators=100, name='Pooled-RF'),
             RandomForestClassifier(n_estimators=100, name='RF-STL'),
             GradientBoostingClassifier(n_estimators=100, name='GBC-STL')]

  # list of metrics to measure method's performance
  # see list of available metrics in utils/performance_metrics.py
  metrics = ['accuracy']

  exp_folder = __file__.strip('.py')
  exp = ModelTraining(exp_folder)
  exp.execute(dataset, methods, metrics, nb_runs=3)
  exp.generate_report()
