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

# folder containing the preprocessed (pkl files)
path_to_data = '/data/SEER/preprocessed_data_2015'

# path to data files
if username in 'goncalves1':

    if 'Darwin' in os_system:  # local - only works for macOS users
        # path to data files
        path_to_data = ('/Users/goncalves1/Documents/projects/'
                        'mtl4c/data/preprocessed_data_2015')

        path_to_data_sso = ('/Users/goncalves1/Documents/projects/mtl4c/repo/data/CRN/processed_data')

        # path to definition files: datasets and experiments json files
        path_to_json = ('/Users/goncalves1/Documents/projects/mtl4c/'
                        'repo/component1/predictive_models/'
                        'auxiliary_files/json_files/')

        # path_to_def = {'dataset': ('/Users/goncalves1/Documents/projects/mtl4c/'
        #                            'repo/component1/predictive_models/'
        #                            'auxiliary_files/json_files/datasets.json'),
        #                'experiment': ('/Users/goncalves1/Documents/projects/mtl4c/'
        #                               'repo/component1/predictive_models/'
        #                               'auxiliary_files/json_files/experiments.json')}

    else:  # on selene
        # path to definition files: datasets and experiments json files
        # path_to_def = {'dataset': ('/home/goncalves1/projects/mtl4c/'
        #                            'repo/component1/predictive_models/'
        #                            'auxiliary_files/json_files/datasets.json'),
        #                'experiment': ('/home/goncalves1/projects/mtl4c/'
        #                               'repo/component1/predictive_models/'
        #                               'auxiliary_files/json_files/'
        #                               'experiments.json')}

        path_to_json = ('/home/goncalves1/projects/mtl4c/'
                        'repo/component1/predictive_models/'
                        'auxiliary_files/json_files/')

elif username in 'widemann1':

    if 'Darwin' in os_system:
        path_to_data = '/Users/widemann1/Documents/cancer/component1/data/preprocessed_data_2015/'
        path_to_data_sso = ('/Users/widemann1/Documents/cancer/data/CRN/Mari')
        # path to definition files: datasets and experiments json files
        path_to_json = ('/Users/widemann1/Documents/cancer/component1'
                       '/predictive_models/'
                       'auxiliary_files/json_files/')
        
        crn_mari_path = '/Users/widemann1/Documents/cancer/data/CRN/Mari'
#        crn_fn = [('CCS','cancerOutcomes/CCS_LLNL_Feb2018.csv'),
#                  ('reg','NorwaySurveyData/regdata.csv'),
#                  ('survey','NorwaySurveyData/surveydata.csv'),
#                  ('surveyAndScreening','NorwaySurveyData/SurveyandScreeningData.csv'),
#                  ('sso','NorwaySurveyData/sso.static.v1.csv')]
#        crn_files = dict()
#        for k,v in crn_fn:
#            crn_files[k] = os.path.join(crn_mari_path,v) 
#        
        # path to definition files: datasets and experiments json files
        path_to_def = {'dataset': ('/Users/widemann1/Documents/cancer/component1/'
                                   'predictive_models/auxiliary_files/json_files/'
                                   'datasets.json'),
                       'experiment': ('/Users/widemann1/Documents/cancer/component1/'
                                      'predictive_models/auxiliary_files/json_files/'
                                      'experiments.json')}
    else:
#        path_to_data = '/home/widemann1/cancer/component1/data/preprocessed_pkl/'
        
        # path to definition files: datasets and experiments json files
        path_to_def = {'dataset': ('/home/widemann1/cancer/component1/'
                                   'predictive_models/auxiliary_files/json_files/'
                                   'datasets.json'),
                       'experiment': ('/home/widemann1/cancer/component1/'
                                      'predictive_models/auxiliary_files/json_files/'
                                      'experiments.json')}

elif username in 'soper3':

    if 'Darwin' in os_system:
        path_to_data  = '/Users/soper3/Documents/MTL4C/data/preprocessed_pkl'
    
        path_to_def = {'dataset': ('/Users/soper3/Documents/MTL4C/'
                                   'repos/component1/predictive_models/auxiliary_files/json_files/'
                                   'datasets.json'),
                       'experiment': ('/Users/soper3/Documents/MTL4C/'
                                      'repos/component1/predictive_models/auxiliary_files/json_files/'
                                      'experiments.json')}
    else:
        # Braden: set your path on selene here
        path_to_def = {'dataset': ('/home/soper3/component1/predictive_models/'
                                   'auxiliary_files/json_files/datasets.json'),
                       'experiment': ('/home/soper3/component1/predictive_models/'
                                      'auxiliary_files/json_files/experiments.json')}

elif username in 'ray34':

    if 'Darwin' in os_system:
        path_to_data  = '/Users/ray34/Documents/Priyadip_work/MTL4C_git/data/preprocessed_data_2015'
    
        path_to_def = {'dataset': ('/Users/ray34/Documents/Priyadip_work/MTL4C_git/component1/predictive_models/auxiliary_files/json_files/datasets.json'),
                       'experiment': ('/Users/ray34/Documents/Priyadip_work/MTL4C_git/component1/predictive_models/auxiliary_files/json_files/experiments.json')}

    else:
#        path_to_data = '/home/widemann1/cancer/component1/data/preprocessed_pkl/'
        
        # path to definition files: datasets and experiments json files
        path_to_def = {'dataset': ('/home/ray34/component1/predictive_models/auxiliary_files/json_files/datasets.json'),
                       'experiment': ('/home/ray34/component1/predictive_models/auxiliary_files/json_files/experiments.json')}
                       
                   
elif username in 'deoliveirasa1':
    path_to_data = "/Users/deoliveirasa1/Documents/projects/MTL4C/data/SEER/preprocessed_2014"
    path_to_def = {'dataset': ('/Users/deoliveirasa1/Documents/projects/MTL4C/code/component1/predictive_models/'
                               'auxiliary_files/json_files/datasets.json'),
                    'experiment': ('/Users/deoliveirasa1/Documents/projects/MTL4C/code/component1/predictive_models/'
                                   'auxiliary_files/json_files/experiments.json')}

elif username is 'abdulla1':
    path_to_def = {'dataset': ('/home/abdulla1/projects/component1/predictive_models/'
                               'auxiliary_files/json_files/datasets.json'),
                    'experiment': ('/home/abdulla1/projects/component1/predictive_models/'
                                   'auxiliary_files/json_files/experiments.json')}

else:
    print('Warning: username not in config.py')
