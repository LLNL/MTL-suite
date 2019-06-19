# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-06-17 15:34:50
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-06-18 12:15:04
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
        self.sample_pointer = 0  # point to the first task
        self.batch_size = batch_size
        self.validation_split = validation_split

        # save a part of the training data for validation
        idx = np.random.permutation(range(x.shape[0]))
        ntrain = int(np.floor(x.shape[0] * (1 - self.validation_split)))
        self.x_train = x[idx[:ntrain], :].copy()
        self.x_val = x[idx[ntrain:], :].copy()

        # print('Number of training samples: {}'.format(self.x_train.shape[0]))
        # print('Number of validation samples: {}'.format(self.x_val.shape[0]))

        self.y_train = y[idx[:ntrain], :].copy()
        self.y_val = y[idx[ntrain:], :].copy()

    def next_batch(self):
        """
        Pick up the next minibatch from the entire dataset.
        """
        if self.sample_pointer >= self.x_train.shape[0]:
            return None

        beg = self.sample_pointer
        end = np.minimum(self.sample_pointer + self.batch_size,
                         self.x_train.shape[0])
        x_mbatch = self.x_train[beg:end, :].copy()
        y_mbatch = self.y_train[beg:end, :].copy()
        self.sample_pointer += self.batch_size

        return {'x': x_mbatch, 'y': y_mbatch}  # , 'w': w_mbatch}

    def set_new_epoch(self):
        '''
        Shuffle the training data for a new epoch.
        '''
        self.sample_pointer = 0
        idx = np.random.permutation(range(self.x_train.shape[0]))
        self.x_train = self.x_train[idx, :].copy()
        self.y_train = self.y_train[idx, :].copy()

    def get_training_steps(self):
        """ Compute the number of training steps inside of an epoch. """
        nb_samples = self.x_train.shape[0]
        return int(np.ceil(nb_samples / self.batch_size))


class DNNClassifierPooled(BaseMTLEstimator):
    """
    Implement a Pooled Deep Neural Network for classification.
    """

    def __init__(self, batch_size=100, nb_epochs=100,
                 fit_intercept=False, normalize=True, name='DNNC-P'):
        """ Initialize object with the informed hyper-parameter values.
        Args:
            batch_size (int): mini-batch size.
            nb_epochs (int): number of training epochs.
            fit_intercept (bool): fit intercept or not
            normalize (bool): normalize the data before training or not
            name (str): string to represent the method
        """
        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        assert batch_size >= 0
        assert nb_epochs >= 0

        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.patience = 10
        self.model = None

    def _fit(self, x, y, **kwargs):
        """
        Train model on given data x and y.
        Args:
            x (numpy.array): input data matrix (covariates).
            y (numpy.array): label vector (outcome).
        Returns:
            None.
        """
        self.ntasks = len(x)  # get number of tasks
        self.ndimensions = x[0].shape[1]  # get problem dimension

        for t in range(self.ntasks):
            x[t] = x[t].astype(np.float64)
            y[t] = y[t].astype(np.int32).ravel()

        # pooled dataset: data from all tasks
        xpooled = np.row_stack(x)
        ypooled = np.concatenate(y)[:, np.newaxis]

        data_gen = Data_Generator(xpooled, ypooled,
                                  batch_size=self.batch_size,
                                  validation_split=0.15)

        self.graph = tf.Graph()
        with self.graph.as_default():
            # tf Graph inputs - this is a single input model
            self.X = tf.placeholder(tf.float32, [None, self.ndimensions], name='X')
            self.Y = tf.placeholder(tf.float32, [None, 1], name='Y')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            (heads_losses, train_ops,
             predict_op, summary_op) = self.__build_model(self.X,
                                                          self.Y,
                                                          # self.W,
                                                          self.keep_prob)
            self.predict_op = predict_op

            # Initialize the variables (i.e. assign their default value)
            init = tf.global_variables_initializer()

            # 'Saver' op to save and restore all the variables
            self.saver = tf.train.Saver()
            # print('# of params: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        # Start training
        with tf.Session(graph=self.graph) as sess:
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
                # start a new epoch: shuffle the data
                data_gen.set_new_epoch()

                # Loop over all batches
                while True:
                    batch = data_gen.next_batch()
                    if batch is None:
                        break

                    # Run optimization op (backprop), cost op
                    # (to get loss value) and summary nodes
                    _, loss, summary = sess.run([train_ops, heads_losses,
                                                 summary_op],
                                                feed_dict={self.X: batch['x'],
                                                           self.Y: batch['y'],
                                                           self.keep_prob: 0.9})
                    # Write logs at every iteration
                    train_writer.add_summary(summary, cont)
                    # Compute average loss over all training steps of that
                    # specific epoch
                    hist_train[epoch] += loss / data_gen.get_training_steps()
                    cont += 1

                # check how model is doing in the validation set
                summary, loss_val = sess.run([summary_op,
                                              heads_losses],
                                             feed_dict={self.X: data_gen.x_val,
                                                        self.Y: data_gen.y_val,
                                                        self.keep_prob: 1.0})
                # print('[{}] Validation loss: {}'.format(epoch, loss_val))
                hist_val[epoch] += loss_val
                test_writer.add_summary(summary, epoch)

                # if new best model on validation set is found, save it into
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
                else:
                    fname = os.path.join(self.output_directory,
                                         'best_mh_model.mdl')
                    _ = self.saver.save(sess, fname)
                    self.logger.info(("Best model found! "
                                      "Saved at: best_mh_model.mdl"))

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
            self.logger.info('Training process finalized.')

    def _predict(self, x, **kwargs):
        """ Predict regression value for the input x.
        Args:
            x (pandas.DataFrame): input data matrix.
        Returns:
            numpy.array with the predicted values.
        """
        # convert to numpy array
        with tf.Session(graph=self.graph) as session:
            # restore the model
            fname = os.path.join(self.output_directory,
                                 'best_mh_model.mdl')
            self.saver.restore(session, fname)
            nb_tasks = len(x)  # number of tasks
            yhat = [None] * nb_tasks
            for t in range(nb_tasks):
                yhat[t] = session.run(self.predict_op,
                                      feed_dict={self.X: x[t],
                                                 self.keep_prob: 1.0})
        return yhat

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        batch_sizes = [16, 32, 64, 100, 200]
        for bt_size in batch_sizes:
            yield {'batch_size': bt_size}

    def __build_model(self, X, Y, keep_prob,
                      larger_layer_size=10, smaller_layer_size=5):
        """ Define DNN architecture. This is just an example.
        You should change it for your own problem. The example here
        consist of a 2-layer neural network with relu activation
        and dropout. """

        # ####### Layer #######
        layer1 = tf.layers.dense(inputs=X,
                                 units=larger_layer_size,
                                 activation=tf.nn.sigmoid,
                                 name='layer1')
        # layer1 = tf.layers.batch_normalization(inputs=layer1)
        layer1 = tf.nn.dropout(x=layer1, keep_prob=keep_prob)

        layer2 = tf.layers.dense(inputs=layer1,
                                 units=smaller_layer_size,
                                 activation=tf.nn.sigmoid,
                                 name='layer2')
        layer2 = tf.nn.dropout(x=layer2, keep_prob=keep_prob)

        # layer2 = tf.layers.batch_normalization(inputs=layer2)
        layer3 = tf.layers.dense(inputs=layer2,
                                 units=1,
                                 activation=tf.nn.sigmoid,
                                 name='output')

        # loss function
        final_loss = tf.losses.log_loss(labels=Y,
                                        predictions=layer3)

        # define optimizer
        train_ops = tf.train.AdamOptimizer().minimize(final_loss)

        # for tensorboard monitoring
        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", final_loss)

        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        return final_loss, train_ops, layer3, merged_summary_op

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
