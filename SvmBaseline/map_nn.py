'''
This script is for performing method 2 in the Characterising residual predictive variation in Matched test set.

Here we take COVID+ instances in the matched test set and mapp them to their COVID- nearest neighbours. In this
augmented classification space we recalculate the classification scores of the SVM
'''

import os
import json
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report,
                            confusion_matrix,
                            recall_score,
                            precision_score,
                            make_scorer,
                            RocCurveDisplay,
                            plot_roc_curve,
                            PrecisionRecallDisplay,
                            roc_curve,
                            auc,
                            roc_auc_score,
                            precision_recall_curve,
                            average_precision_score)
from scipy.spatial import distance
from tqdm import tqdm

from svm import RunSvm
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from utils.eval_metrics import AUCSVM

class NNSvm(RunSvm):

    '''
    map covid+ to nn covid- cases in opensmil space
    '''

    def __init__(self, modality, feature_base, results_base, p, pca_red=False):
        super(NNSvm, self).__init__(modality, feature_base)
        self.results_base = results_base + modality
        self.modality = modality
        self.pca_red = pca_red
        if not os.path.exists(self.results_base):
            os.makedirs(self.results_base)
        self.metrics = {}
        self.test_y = self.matched_test_y
        self.test_names = self.matched_test_names
        self.test_X = self.matched_test_X
        #we have already performed our hyp search for svm. See svm.py
        self.train_y = self.concat(self.matched_train_y, self.matched_devel_y)
        self.train_names = self.concat(self.matched_train_names, self.matched_devel_names)
        self.train_X = self.concat(self.matched_train_X, self.matched_devel_X)
        
        if not self.pca_red: 
            self.scale_data()
            curated_test_X, curated_test_y, curated_test_names = self.fit_NN(p, self.test_X)
        else:
            print('performing dim reduction before NN calculation')
            self.NN_X = self.dim_red(self.test_X)
            self.scale_data()
            curated_test_X, curated_test_y, curated_test_names = self.fit_NN(p, self.NN_X)

        self.train_svm(
                self.train_X, 
                self.train_y, 
                self.test_X, 
                self.test_y,
                self.test_names,
                curated_test_X, 
                curated_test_y, 
                curated_test_names)
        with open(os.path.join(self.results_base, f'metrics_pca{self.pca_red}.json'), 'w') as f:
            json.dump(self.metrics, f)

    def load_data(self, feature_folder):
        
        matched_train_file = os.path.join(feature_folder,
                                     self.modality,
                                     'matched_train.csv')
        matched_devel_file = os.path.join(feature_folder,
                                     self.modality,
                                     'matched_validation.csv')
        matched_test_file = os.path.join(feature_folder,
                                        self.modality,
                                         'matched_test.csv')
        label_index = -1

        print('loading csvs')

        matched_train_df = pd.read_csv(matched_train_file, skiprows=6379, header=None)
        self.matched_train_X = matched_train_df.values[:, 1:label_index].astype(np.float32)
        self.matched_train_y = matched_train_df.values[:, label_index].astype(str)
        self.matched_train_names = matched_train_df.values[:,0]
        
        matched_devel_df = pd.read_csv(matched_devel_file, skiprows=6379, header=None)
        self.matched_devel_X = matched_devel_df.values[:, 1:label_index].astype(np.float32)
        self.matched_devel_y = matched_devel_df.values[:, label_index].astype(str)
        self.matched_devel_names = matched_devel_df.values[:,0]
        
        matched_test_df = pd.read_csv(matched_test_file, skiprows=6379, header=None)
        self.matched_test_X = matched_test_df.values[:, 1:label_index].astype(np.float32)
        self.matched_test_y = matched_test_df.values[:, label_index].astype(str)        
        self.matched_test_names = matched_test_df.values[:, 0]

    def concat(self, train, devel):
        return np.append(train, devel, axis=0)

    def scale_data(self):
        #first fit scaler to train
        scaler = StandardScaler()
        self.train_X = scaler.fit_transform(self.train_X)
        # then transform test using the scale fitted to train
        self.test_X = scaler.transform(self.test_X)

    def fit_NN(self, p, test_X):
        '''
        fit the NN unsupervised algorithm
        '''

        covid_neg_ind = np.where(self.test_y == 'Negative')[0]
        covid_neg = test_X[covid_neg_ind]
        covid_neg_names = self.test_names[covid_neg_ind]
        covid_neg_y = self.test_y[covid_neg_ind]
        assert not any([x == 'Positive' for x in covid_neg_y])
        covid_pos_ind = np.where(self.test_y == 'Positive')[0]
        covid_pos = test_X[covid_pos_ind]
        covid_pos_names = self.test_names[covid_pos_ind]
        covid_pos_y = self.test_y[covid_pos_ind]
        assert not any([x == 'Negative' for x in covid_pos_y])
        #if pca_red we calculate the nearest neighbours in pca space. We then map the positive instances
        # to the covid negative instances but in the opensmile space.
        nbrs = NearestNeighbors(n_neighbors=1, p=p).fit(covid_neg)
        distances, indices = nbrs.kneighbors(covid_pos)

        plt.figure()
        plt.yscale('log', nonposy='clip')
        plt.hist(distances, bins=100, color='b')
        plt.title('Histogram of distance to NN')
        plt.xlabel('Euclidean Distance in normalised space')
        plt.savefig(f'{self.results_base}_distance_histogram_pca{self.pca_red}.png', bbox_inches='tight')
        plt.close()
        plt.figure()
        plt.hist(indices, bins=100, color='g')
        plt.title('Histogram of which Negative instances positive instances were mapped to')
        plt.xlabel('Negative index')
        plt.savefig(f'{self.results_base}_indices_histogram_pca{self.pca_red}.png', bbox_inches='tight')

        if self.pca_red:
            # we are still training and classifying in the opensmile space and not pca
            open_covid_neg = self.test_X[covid_neg_ind]
            open_covid_pos = self.test_X[covid_pos_ind]
            pos_map_neg = self.map_to_nn(
                    indices,
                    open_covid_neg,
                    covid_neg_names,
                    covid_neg_y,
                    open_covid_pos,
                    covid_pos_names,
                    covid_pos_y)
            X = np.append(open_covid_neg, pos_map_neg, axis=0)
            y = np.append(covid_neg_y, covid_pos_y, axis=0)
            names = np.append(covid_neg_names, covid_pos_names, axis=0)
            return X, y, names
        pos_map_neg = self.map_to_nn(
                indices,
                covid_neg,
                covid_neg_names,
                covid_neg_y,
                covid_pos,
                covid_pos_names,
                covid_pos_y)
        X = np.append(covid_neg, pos_map_neg, axis=0)
        y = np.append(covid_neg_y, covid_pos_y, axis=0)
        names = np.append(covid_neg_names, covid_pos_names, axis=0)
        return X, y, names

    def map_to_nn(self,
                indices,
                covid_neg,
                covid_neg_names,
                covid_neg_y,
                covid_pos,
                covid_pos_names,
                covid_pos_y):
        curated_test = covid_pos 
        for pos_i, nn_i in enumerate(indices):
            print(pos_i, '   ', nn_i)
            # but we keep the name and the label for covid the same!
            curated_test[pos_i,:] = covid_neg[nn_i[0],:]

        return curated_test

    def dim_red(self, X):
        '''
        Allow us to first reduce the dim of the opensmile space before performing the NN calculation
        '''
        pca = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=50))])
        pca.fit(np.nan_to_num(X))
        return pca.transform(np.nan_to_num(X))

    def train_svm(self, train_X, train_y, test_X, test_y, test_names, curated_test_X, curated_test_y, curated_test_names):
        svm = make_pipeline(
                    LinearSVC(
                        random_state=42,
                        class_weight='balanced',
                        max_iter=10000,
                        loss='squared_hinge',
                        C=0.0001))
        svm.fit(train_X, train_y)
        preds = svm.predict(test_X)
        self.run_test(
                svm,
                train_X,
                train_y,
                test_X,
                test_y,
                test_names,
                'original',
                None,
                self.results_base)

        preds = svm.predict(curated_test_X)
        self.run_test(
                svm,
                train_X,
                train_y,
                curated_test_X,
                curated_test_y,
                curated_test_names,
                'curated',
                None,
                self.results_base)

    def run_test(self,
                estimator,
                train_X,
                train_y,
                test_X:np.array,
                test_y:np.array,
                test_names,
                test_name:str,
                params,
                results_folder):
        '''
        Run prediction and evaluation metrics for a test set: X, y
        '''
        test_X = np.nan_to_num(test_X)
        preds = estimator.predict(test_X)
        uar = recall_score(test_y, preds, average='macro')
        cm = confusion_matrix(test_y, preds)
        auc_metrics = AUCSVM(estimator, test_X, test_y)

        #plot_roc_curve(estimator, test_X, test_y, pos_label='Positive')
        fig, prec, rec, pr_auc = auc_metrics.PR_AUC()
        fig, fpr, tpr, roc_auc = auc_metrics.ROC_AUC()

        self.metrics[test_name] = {}
        self.metrics[test_name]['precision'] = prec.tolist()
        self.metrics[test_name]['recall'] = rec.tolist()
        self.metrics[test_name]['fpr'] = fpr.tolist()
        self.metrics[test_name]['tpr'] = tpr.tolist()
        self.metrics[test_name]['pr_auc'] = pr_auc
        self.metrics[test_name]['roc_auc'] = roc_auc
        self.metrics[test_name]['uar'] = uar
        self.metrics[test_name]['cm'] = cm.tolist()
        print(f'{test_name}\nUAR: {uar}\nROC-AUC: {roc_auc}\n{classification_report(test_y, preds)}\nConfusion Matrix:\n\n{cm}')

        df_predictions = pd.DataFrame({'filename': test_names.tolist(), 'prediction': preds.tolist()})
        df_predictions.to_csv(os.path.join(results_folder, f'{test_name}.predictions.csv'), index=False)

if __name__ == '__main__':

    feature_base ='./features/opensmile_final/'
    results_base= './results_september_2022/'


    n = NNSvm('audio_sentence_url', feature_base, results_base, p=1, pca_red=True)
