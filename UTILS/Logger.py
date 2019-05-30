#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:23:20 2018

@author: widemann1, goncalves1
"""

import os
import logging
from logging.handlers import RotatingFileHandler


class Logger(object):
    """
    Log handling class.

    Attributes:
        log_directory: path to directory where log file will be saved.
        log_filename: log filename (only name no path).
        log: python's logging object
    """
    def __init__(self):
        """ Initialize class's attributes. """
        self.log_directory = None
        self.log_filename = None
        self.log = None

    def setup_logger(self, filename='system.log', stream=False,
                     maxBytes=20, bkpCount=5):
        """
            Creates logger for program output.
        Args:
            filename (str): log filename (not path, only filename).
            stream (bool): flag if information will be printed on the console.
            maxbytes (int): maximum number of bytes a log file should have
                            before roll-over.
            backupCount (int): after file exceds maxbytes how many files
                                should be create.
        """
        # set log filename
        if filename.endswith('.log'):
            self.log_filename = filename
        else:
            self.log_filename = '{}.log'.format(filename)

        # if log file already exist: delete it
        if os.path.exists(os.path.join(self.log_directory, self.log_filename)):
            os.remove(os.path.join(self.log_directory, self.log_filename))

        # define output format
        fmt = ('%(asctime)s - %(levelname)s -'
               '%(module)s.%(funcName)s : %(message)s')
        datefmt = '%m/%d/%Y %I:%M:%S %p'
        formatter = logging.Formatter(fmt, datefmt=datefmt)

        # whole path to log file
        log_path_fname = os.path.join(self.log_directory, self.log_filename)

        # create logging element
        logging.basicConfig(format=fmt,
                            filename=log_path_fname,
                            datefmt=datefmt,
                            level=logging.INFO)

        log = logging.getLogger(name=log_path_fname)
        log.handlers = []

        # if log reaches maxBytes, it gets “rolled over” and
        # a new log file is created
        fh = RotatingFileHandler(log_path_fname, maxBytes, bkpCount)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        log.addHandler(fh)

        # also print logs on the console
        if stream:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            ch.setLevel(logging.INFO)
            log.addHandler(ch)

        # set logging object
        self.log = log

    def info(self, output_str):
        """
        Writes content into log file at a INFO level.

        Args:
            output_str (str): string to be written in the log file.
        """
        if self.log is None:
            raise ValueError('Must setup logger before call info.')
        else:
            self.log.info(output_str)

    def set_path(self, path):
        """ Set output directory path.
        Args:
            path (str): path to output directory
        """
        self.log_directory = path
        # make sure output path exists
        # if it doesn't, then throw an error to the user
        # the user must create a log directory before calling logger
        assert os.path.exists(path)
