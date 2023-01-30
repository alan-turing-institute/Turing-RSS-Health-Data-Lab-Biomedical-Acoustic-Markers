'''
author: Harry Coppock
contact: harry.coppock@imperial.ac.uk
file: symptoms_svm.py
notes: How well does a symptoms classifier (trained on covid neg symptoms when evaluated on COVID predicition.?
'''
from sklearn.svm import LinearSVC
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.model_selection import PredefinedSplit, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (classification_report,
                                confusion_matrix,
                                recall_score,
                                make_scorer,
                                RocCurveDisplay,
                                plot_roc_curve,
                                plot_precision_recall_curve)
from joblib import Parallel, delayed
import pandas as pd
import os, yaml
import json
import sys
import scipy
from scipy.io.arff import loadarff
import numpy as np
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from utils.eval_metrics import AUCSVM

from svm import RunSvm

POSSIBLE_MODALITIES = ['audio_sentence_url',
                       'audio_cough_url',
                       'audio_three_cough_url',
                        'audio_ha_sound_url'
                        ]

class SympSvm(RunSvm):
    def __init__(self, modality, feature_folder, symptom_folder):
        self.modality = modality
        self.metrics = {}
        print('Starting formatting features')
        self.load_data(feature_folder, symptom_folder)

    def load_data(self, feature_folder, symptom_folder):
        '''
        overwrite RunSvm load_data method to load in new train, val and naive train files
        '''
        
        matched_test_file = os.path.join(feature_folder,
                                        self.modality,
                                         'matched_test.csv')
        train_file = os.path.join(symptom_folder,
                                     self.modality.replace('audio_', ''),
                                     'train.csv')
        devel_file = os.path.join(symptom_folder,
                                        self.modality.replace('audio_', ''),
                                        'val.csv')
        test_file = os.path.join(feature_folder,
                                        self.modality,
                                         'test.csv')
        long_test_file = os.path.join(feature_folder,
                                        self.modality,
                                         'long_test.csv')
        long_matched_test_file = os.path.join(feature_folder,
                                        self.modality,
                                         'long_matched_test.csv')

        #naive
        naive_train_file = os.path.join(symptom_folder,
                                        self.modality.replace('audio_', ''),
                                     'naive_train.csv')
        naive_devel_file = os.path.join(symptom_folder,
                                        self.modality.replace('audio_', ''),
                                        'naive_validation.csv')
        naive_test_file = os.path.join(feature_folder,
                                        self.modality,
                                         'naive_test.csv')
        label_index = -1

        print('loading csvs')

        train_df = pd.read_csv(train_file, skiprows=6379, header=None)
        self.train_X = train_df.values[:, 1:label_index].astype(np.float32)
        self.train_y = train_df.iloc[:,label_index].apply(lambda x: 'Positive' if x == 'symptomatic' else 'Negative').values.astype(str)
        self.train_names = train_df.values[:,0]
        
        devel_df = pd.read_csv(devel_file, skiprows=6379, header=None)
        self.devel_X = devel_df.values[:, 1:label_index].astype(np.float32)
        self.devel_y = devel_df.iloc[:,label_index].apply(lambda x: 'Positive' if x == 'symptomatic' else 'Negative').values.astype(str)
        self.devel_names = devel_df.values[:,0]

        test_df = pd.read_csv(test_file, skiprows=6379, header=None)
        self.test_X = test_df.values[:, 1:label_index].astype(np.float32)
        self.test_y = test_df.values[:, label_index].astype(str)        
        self.test_names = test_df.values[:, 0]
        
        long_test_df = pd.read_csv(long_test_file, skiprows=6379, header=None)
        self.long_test_X = long_test_df.values[:, 1:label_index].astype(np.float32)
        self.long_test_y = long_test_df.values[:, label_index].astype(str)        
        self.long_test_names = long_test_df.values[:, 0]
    
        long_matched_test_df = pd.read_csv(long_matched_test_file, skiprows=6379, header=None)
        self.long_matched_test_X = long_matched_test_df.values[:, 1:label_index].astype(np.float32)
        self.long_matched_test_y = long_matched_test_df.values[:, label_index].astype(str)        
        self.long_matched_test_names = long_matched_test_df.values[:, 0]
        
        matched_test_df = pd.read_csv(matched_test_file, skiprows=6379, header=None)
        self.matched_test_X = matched_test_df.values[:, 1:label_index].astype(np.float32)
        self.matched_test_y = matched_test_df.values[:, label_index].astype(str)        
        self.matched_test_names = matched_test_df.values[:, 0]


        #naive splits
        naive_train_df = pd.read_csv(naive_train_file, skiprows=6379, header=None)
        self.naive_train_X = naive_train_df.values[:, 1:label_index].astype(np.float32)
        self.naive_train_y = naive_train_df.iloc[:,label_index].apply(lambda x: 'Positive' if x == 'symptomatic' else 'Negative').values.astype(str)
        self.naive_train_names = naive_train_df.values[:,0]
        naive_devel_df = pd.read_csv(naive_devel_file, skiprows=6379, header=None)
        self.naive_devel_X = naive_devel_df.values[:, 1:label_index].astype(np.float32)
        self.naive_devel_y = naive_devel_df.iloc[:,label_index].apply(lambda x: 'Positive' if x == 'symptomatic' else 'Negative').values.astype(str)
        self.naive_devel_names = naive_devel_df.values[:,0]
        naive_test_df = pd.read_csv(naive_test_file, skiprows=6379, header=None)
        self.naive_test_X = naive_test_df.values[:, 1:label_index].astype(np.float32)
        self.naive_test_y = naive_test_df.values[:, label_index].astype(str)        
        self.naive_test_names = naive_test_df.values[:, 0]


if __name__ == '__main__':


    params = {'ci_interval': 0.95, 'bootstrap_iterations': 1000}
    feature_base ='./features/opensmile_final/'
    symptoms_base = './features/opensmile_symptoms_clf/'
    for modality in POSSIBLE_MODALITIES:
        print('*'*20)
        print(f'Starting with modality: {modality}')
        print('*'*20)
        svm_instance = SympSvm(modality, feature_base, symptoms_base)
        for train_scheme in ['standard', 'naive']:#, 'matched']:#, 'original', 'original_matched']:
            result_base = f'./results_septemeber_2022/symptoms/{modality}/{train_scheme}/'
            print(train_scheme)
            if not os.path.exists(result_base):
                os.makedirs(result_base)

            if train_scheme == 'standard':
                train_X=svm_instance.train_X
                train_y=svm_instance.train_y
                train_names=svm_instance.train_names
                devel_X=svm_instance.devel_X
                devel_y=svm_instance.devel_y
                devel_names=svm_instance.devel_names
                test_X=svm_instance.test_X
                test_y=svm_instance.test_y
                test_names=svm_instance.test_names
                matched_test_X=svm_instance.matched_test_X
                matched_test_y=svm_instance.matched_test_y
                matched_test_names=svm_instance.matched_test_names
                long_test_X=svm_instance.long_test_X
                long_test_y=svm_instance.long_test_y
                long_test_names=svm_instance.long_test_names
                long_matched_test_X=svm_instance.long_matched_test_X
                long_matched_test_y=svm_instance.long_matched_test_y
                long_matched_test_names=svm_instance.long_matched_test_names
            elif train_scheme == 'naive':
                train_X=svm_instance.naive_train_X
                train_y=svm_instance.naive_train_y
                train_names=svm_instance.naive_train_names
                devel_X=svm_instance.naive_devel_X
                devel_y=svm_instance.naive_devel_y
                devel_names=svm_instance.naive_devel_names
                test_X=svm_instance.naive_test_X
                test_y=svm_instance.naive_test_y
                test_names=svm_instance.naive_test_names

            elif train_scheme == 'matched':
                train_X=svm_instance.matched_train_X
                train_y=svm_instance.matched_train_y
                train_names=svm_instance.matched_train_names
                devel_X=svm_instance.matched_devel_X
                devel_y=svm_instance.matched_devel_y
                devel_names=svm_instance.matched_devel_names
                test_X=svm_instance.test_X
                test_y=svm_instance.test_y
                test_names=svm_instance.test_names
                matched_test_X=svm_instance.matched_test_X
                matched_test_y=svm_instance.matched_test_y
                matched_test_names=svm_instance.matched_test_names
                long_test_X=svm_instance.long_test_X
                long_test_y=svm_instance.long_test_y
                long_test_names=svm_instance.long_test_names
                long_matched_test_X=svm_instance.long_matched_test_X
                long_matched_test_y=svm_instance.long_matched_test_y
                long_matched_test_names=svm_instance.long_matched_test_names
            else:
                raise f'{train_scheme} is not a valid train scheme'


            svm_instance.run_svm(
                feature_base,
                result_base,
                params,
                train_scheme=train_scheme,
                train_X=train_X,
                train_y=train_y,
                train_names=train_names,
                devel_X=devel_X,
                devel_y=devel_y,
                devel_names=devel_names,
                test_X=test_X,
                test_y=test_y,
                test_names=test_names,
                matched_test_X=matched_test_X,
                matched_test_y=matched_test_y,
                matched_test_names=matched_test_names,
                long_test_X=long_test_X,
                long_test_y=long_test_y,
                long_test_names=long_test_names,
                long_matched_test_X=long_matched_test_X,
                long_matched_test_y=long_matched_test_y,
                long_matched_test_names=long_matched_test_names)
