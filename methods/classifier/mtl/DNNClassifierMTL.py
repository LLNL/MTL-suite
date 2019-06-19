# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-06-17 15:14:42
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-06-18 14:13:48
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
from ..base import BaseMTLEstimator


class Data_Generator(object):

    def __init__(self, x, y, batch_size=100, validation_split=0.2):
        """
        x: list of numpy-array covariates - one for each task
        y: list of numpy-array outputs - one for each task
        """
        self.task_pointer = 0  # point to the first task
        self.nb_tasks = len(x)
        self.batch_size = batch_size
        self.validation_split = validation_split

        # make sure all tasks have data
        for k in range(self.nb_tasks):
            assert x[k].shape[0] == y[k].shape[0]
            assert x[k].shape[0] > 0

        # separate a part of the training data for validation
        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()
        for k in range(self.nb_tasks):
            idx = np.random.permutation(range(x[k].shape[0]))
            ntrain = int(np.floor(x[k].shape[0] * (1 - self.validation_split)))
            self.x_train.append(x[k][idx[0:ntrain], :])
            self.x_val.append(x[k][idx[ntrain:], :])
            if len(y[k].shape) < 2:
                self.y_train.append(y[k][idx[0:ntrain], np.newaxis])
                self.y_val.append(y[k][idx[ntrain:], np.newaxis])

    def next_batch(self):
        """
        Pick a minibatch from a specific task.
        Circulates over all tasks.
        """
        # number of data points in the self.task_pointer-th task
        data_idx = range(self.x_train[self.task_pointer].shape[0])
        # if the amount of data in that specific task is smaller than
        # the batch size, sample with replacement
        replacement = False
        if len(data_idx) < self.batch_size:
            replacement = True
        sample = np.random.choice(data_idx,
                                  size=self.batch_size,
                                  replace=replacement)
        x_mbatch = self.x_train[self.task_pointer][sample, :]
        y_mbatch = self.y_train[self.task_pointer][sample, :]
        task_id = self.task_pointer

        # circulates over all tasks
        # in this way we make sure the same amount of data for all tasks
        # are being used in the training
        if self.task_pointer + 1 == self.nb_tasks:
            self.task_pointer = 0
        else:
            self.task_pointer += 1
        return task_id, x_mbatch, y_mbatch

    def get_training_steps(self):
        nb_samples = [self.x_train[i].shape[0] for i in range(self.nb_tasks)]
        return int(np.floor(np.array(nb_samples).max() / self.batch_size))


class DNNClassifierMTL(BaseMTLEstimator):
    """
    Implement a Multi-Head MTL Deep Neural Network classifier.

    Attributes:
        batch_size (int): size of the mini-batch.
        nb_epochs (int): number of epochs for training
    """

    def __init__(self, batch_size=100, nb_epochs=100, name='DNN',
                 fit_intercept=True, normalize=False):
        """ Initialize object with the informed hyper-parameter values.
        Args:
            batch_size (int): mini-batch size.
            nb_epochs (int): number of training epochs.
        """
        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        assert batch_size >= 0
        assert nb_epochs >= 0

        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.patience = 10

        self.model = None

    def __str__(self):
        """ Create a string to be used as reference for a class instance. """
        pars = self.get_params()
        ref = '.'.join(['{}{}'.format(k, pars[k]) for k in pars.keys()])
        ref = ref.replace('_', '').replace('.', '_')
        return '{}_{}'.format(self.__class__.__name__, ref)

    def _fit(self, x, y, **kwargs):
        """
        Train model on given data x and y.
        Args:
            x (numpy.array): input data matrix (covariates).
            y (numpy.array): label vector (outcome).
        Returns:
            None.
        """
        # make sure all tasks' datasets have the same dimension and
        # all x's and y's have the same number of samples

        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default() as graph:
            # tf Graph input
            self.X = tf.placeholder(tf.float32,
                                    [None, x[0].shape[1]],
                                    name='X')
            self.Y = tf.placeholder(tf.float32, [None, 1], name='Y')

            # architecture depend on data dimension
            (heads_losses, train_ops,
             predict_op, summary_op) = self.__multihead_model_fn(self.X,
                                                                 self.Y,
                                                                 len(x))
            self.predict_op = predict_op

            # Initialize the variables (i.e. assign their default value)
            init = tf.global_variables_initializer()

            data_gen = Data_Generator(x, y,
                                      batch_size=self.batch_size,
                                      validation_split=0.2)

            # 'Saver' op to save and restore all the variables
            self.saver = tf.train.Saver()

            # Start training
            with tf.Session(graph=graph) as sess:
                sess.run(init)  # Run the initializer
                train_writer = tf.summary.FileWriter(self.output_directory + '/train',
                                                     sess.graph)
                test_writer = tf.summary.FileWriter(self.output_directory + '/test')

                cont = 0  # optimization steps counter (for tensorboard)
                nonimp_counter = 0  # validation non-improvement counter (early stopping)
                hist_train = np.zeros((self.nb_epochs, ))
                hist_val = np.zeros((self.nb_epochs, ))

                # at each epoch: shuffle data (internally done by Data_Generator)
                for epoch in range(self.nb_epochs):
                    # Loop over all batches
                    for i in range(data_gen.get_training_steps()):
                        k, batch_xs, batch_ys = data_gen.next_batch()
                        # Run optimization op (backprop), cost op
                        # (to get loss value) and summary nodes
                        _, loss, summary = sess.run([train_ops[k], heads_losses[k],
                                                     summary_op],
                                                    feed_dict={self.X: batch_xs,
                                                               self.Y: batch_ys})
                        # Write logs at every iteration
                        train_writer.add_summary(summary, cont)
                        # Compute average loss over all training steps of that
                        # specific epoch
                        hist_train[epoch] += loss / data_gen.get_training_steps()
                        cont += 1

                    # check how model is doing in the validation set
                    for k in range(data_gen.nb_tasks):
                        summary, loss_val = sess.run([summary_op,
                                                      heads_losses[k]],
                                                     feed_dict={self.X: data_gen.x_val[k],
                                                                self.Y: data_gen.y_val[k]})
                        hist_val[epoch] += loss_val / data_gen.nb_tasks
                        test_writer.add_summary(summary, epoch)

                    # if new best model (on valdation set) is found, save it into
                    # a file (for later use)
                    if epoch > 0:
                        nonimp_counter += 1
                        if hist_val[epoch] < hist_val[:epoch].min():
                            # Save model weights to disk
                            fname = os.path.join(self.output_directory,
                                                 'best_mh_model.mdl')
                            _ = self.saver.save(sess, fname)
                            self.logger.info(("Best model found! "
                                              "Saved at: best_mh_model.mdl"))
                            nonimp_counter = 0

                    # Display logs per epoch step
                    msg = ("[Epoch {}] ".format(epoch) +
                           "cost train= {:.9f} ".format(hist_train[epoch]) +
                           "cost val= {:.9f}".format(hist_val[epoch]))
                    self.logger.info(msg)

                    # check if the lastest 'patience'-th models were worse than
                    # the best model found till now (on the validation set),
                    # if so, the model probably started overfitting
                    # and it's time to stop training
                    if nonimp_counter >= self.patience:
                        self.logger.info(('Training stopped as it\'s prone'
                                          ' to overfitting.'))
                        break
            # load best model' weights from file
    #        self.model.load_weights(filename)
            self.logger.info('Training process finalized.')

    def _predict(self, x, **kwargs):
        """ Predict regression value for the input x.
        Args:
            x (pandas.DataFrame): input data matrix.
        Returns:
            numpy.array with the predicted values.
        """
        # convert to numpy array
        with tf.Session(graph=self.tf_graph) as session:
            # restore the model
            fname = os.path.join(self.output_directory,
                                 'best_mh_model.mdl')
            self.saver.restore(session, fname)
            yhat = [None] * len(x)
            for k in range(len(x)):
                yhat[k] = session.run(self.predict_op[k],
                                      feed_dict={self.X: x[k]})

        return yhat

    def set_params(self, params):
        """
        Set hyper-parameters to be used in the execution.
        Args:
            params (dict): dict with hyper-parameter values.
        """
        self.batch_size = params['batch_size']

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'batch_size': self.batch_size}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        batch_sizes = [16, 32, 64, 100, 200]
        for bt_size in batch_sizes:
            yield {'batch_size': bt_size}

    def __multihead_model_fn(self, X, Y, nb_tasks, mode='TRAIN',
                             shared_layer_size=128, heads_layer_size=128):
        """Model function for CNN."""

        # Dense Layer - First Shared Layer
        shared1 = tf.layers.dense(inputs=X, units=shared_layer_size,
                                  activation=tf.nn.relu, name='shared1')

        # Add dropout operation; 0.6 probability that element will be kept
        # dropout1 = tf.layers.dropout(inputs=shared1, rate=0.4,
                                     # training=mode == tf.estimator.ModeKeys.TRAIN)
        # shared2 = tf.layers.dense(inputs=dropout1, units=shared_layer_size,
                                  # activation=tf.nn.sigmoid, name='shared2')

        dropout2 = tf.layers.dropout(inputs=shared1, rate=0.4,
                                     training=mode == tf.estimator.ModeKeys.TRAIN)
        shared3 = tf.layers.dense(inputs=dropout2, units=shared_layer_size,
                                  activation=tf.nn.sigmoid, name='shared3')

        heads_l1 = [None] * nb_tasks
        heads_l2 = [None] * nb_tasks
        heads_out = [None] * nb_tasks
        heads_losses = [None] * nb_tasks
        train_ops = [None] * nb_tasks
        for k in range(nb_tasks):  # for all tasks

            heads_l1[k] = tf.layers.dense(inputs=shared3,
                                          units=heads_layer_size,
                                          activation=tf.nn.sigmoid,
                                          name='lay1_task_{}'.format(k))
            heads_l2[k] = tf.layers.dense(inputs=heads_l1[k],
                                          units=heads_layer_size,
                                          activation=tf.nn.sigmoid,
                                          name='lay2_task_{}'.format(k))
            heads_out[k] = tf.layers.dense(inputs=heads_l2[k],
                                           units=1,
                                           activation=tf.nn.sigmoid,
                                           name='output_{}'.format(k))
            heads_losses[k] = tf.losses.log_loss(labels=Y,
                                                 predictions=heads_out[k])
            train_ops[k] = tf.train.AdamOptimizer().minimize(heads_losses[k])

        # tasks average loss
        mean_loss = tf.reduce_mean(heads_losses)

        # for tensorboard monitoring
        for k in range(nb_tasks):
            # Create a summary to monitor cost tensor
            tf.summary.scalar("loss_task_%s" % (k),
                              heads_losses[k])
        tf.summary.scalar("mean_loss", mean_loss)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        return heads_losses, train_ops, heads_out, merged_summary_op
