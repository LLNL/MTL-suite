#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Definition of paths to data/experiments files and any other global
information that need to be shared among all code files.

Created on Mon Dec 11 10:54:31 2017

@author: goncalves1
"""
import os
import pwd as p
from platform import platform


def get_username():
  return p.getpwuid(os.getuid())[0]


username = get_username()
os_system = platform()

path_to_output = '../outputs'
path_to_datasets = '../../datasets'
