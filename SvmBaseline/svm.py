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

POSSIBLE_MODALITIES = ['audio_sentence_url',
                       'audio_cough_url',
                       'audio_three_cough_url',
                        'audio_ha_sound_url'
                        ]
                        #'combined']
class RunSvm():
    '''
    perform hyp sweep + train + evaluate best svm model on covid data
    Attributes:
        modalalit (str): respiratory audio modality
        feature_folder (str): location of features to load
    '''
    RANDOM_SEED = 42

    GRID = [
        {'scaler': [StandardScaler()],
         'estimator': [LinearSVC(random_state=RANDOM_SEED)],
         'estimator__loss': ['squared_hinge'],
         'estimator__C': np.logspace(-4, -5, num=2),
         'estimator__class_weight': ['balanced'],#, None],
         'estimator__max_iter': [10000]
         }
    ]

    PIPELINE = Pipeline([('scaler', None), ('estimator', LinearSVC())])

    def __init__(self, modality, feature_folder,):
        self.modality = modality
        self.metrics = {
                }
        print('starting formatting features')
        self.load_data(feature_folder)

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
        train_file = os.path.join(feature_folder,
                                     self.modality,
                                     'train.csv')
        devel_file = os.path.join(feature_folder,
                                        self.modality,
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

        #now repeat for the original splits with no asymp in train
        
        original_matched_train_file = os.path.join(feature_folder,
                                     self.modality,
                                     'matched_train_original.csv')
        original_matched_devel_file = os.path.join(feature_folder,
                                     self.modality,
                                     'matched_validation_original.csv')
        original_matched_test_file = os.path.join(feature_folder,
                                        self.modality,
                                         'matched_test_original.csv')
        original_train_file = os.path.join(feature_folder,
                                     self.modality,
                                     'train_original.csv')
        original_devel_file = os.path.join(feature_folder,
                                        self.modality,
                                        'val_original.csv')
        original_test_file = os.path.join(feature_folder,
                                        self.modality,
                                         'test_original.csv')
        #naive
        naive_train_file = os.path.join(feature_folder,
                                     self.modality,
                                     'naive_train.csv')
        naive_devel_file = os.path.join(feature_folder,
                                        self.modality,
                                        'naive_validation.csv')
        naive_test_file = os.path.join(feature_folder,
                                        self.modality,
                                         'naive_test.csv')
        label_index = -1

        print('loading csvs')

        train_df = pd.read_csv(train_file, skiprows=6379, header=None)
        self.train_X = train_df.values[:, 1:label_index].astype(np.float32)
        self.train_y = train_df.values[:, label_index].astype(str)
        self.train_names = train_df.values[:,0]
        
        devel_df = pd.read_csv(devel_file, skiprows=6379, header=None)
        self.devel_X = devel_df.values[:, 1:label_index].astype(np.float32)
        self.devel_y = devel_df.values[:, label_index].astype(str)
        self.devel_names = devel_df.values[:,0]
        
        matched_train_df = pd.read_csv(matched_train_file, skiprows=6379, header=None)
        self.matched_train_X = matched_train_df.values[:, 1:label_index].astype(np.float32)
        self.matched_train_y = matched_train_df.values[:, label_index].astype(str)
        self.matched_train_names = matched_train_df.values[:,0]
        
        matched_devel_df = pd.read_csv(matched_devel_file, skiprows=6379, header=None)
        self.matched_devel_X = matched_devel_df.values[:, 1:label_index].astype(np.float32)
        self.matched_devel_y = matched_devel_df.values[:, label_index].astype(str)
        self.matched_devel_names = matched_devel_df.values[:,0]

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

        #original splits:
        original_train_df = pd.read_csv(original_train_file, skiprows=6379, header=None)
        self.original_train_X = original_train_df.values[:, 1:label_index].astype(np.float32)
        self.original_train_y = original_train_df.values[:, label_index].astype(str)
        self.original_train_names = original_train_df.values[:,0]
        
        original_devel_df = pd.read_csv(original_devel_file, skiprows=6379, header=None)
        self.original_devel_X = original_devel_df.values[:, 1:label_index].astype(np.float32)
        self.original_devel_y = original_devel_df.values[:, label_index].astype(str)
        self.original_devel_names = original_devel_df.values[:,0]
        
        original_test_df = pd.read_csv(original_test_file, skiprows=6379, header=None)
        self.original_test_X = original_test_df.values[:, 1:label_index].astype(np.float32)
        self.original_test_y = original_test_df.values[:, label_index].astype(str)        
        self.original_test_names = original_test_df.values[:, 0]
        
        original_matched_train_df = pd.read_csv(original_matched_train_file, skiprows=6379, header=None)
        self.original_matched_train_X = original_matched_train_df.values[:, 1:label_index].astype(np.float32)
        self.original_matched_train_y = original_matched_train_df.values[:, label_index].astype(str)
        self.original_matched_train_names = original_matched_train_df.values[:,0]
        
        original_matched_devel_df = pd.read_csv(original_matched_devel_file, skiprows=6379, header=None)
        self.original_matched_devel_X = original_matched_devel_df.values[:, 1:label_index].astype(np.float32)
        self.original_matched_devel_y = original_matched_devel_df.values[:, label_index].astype(str)
        self.original_matched_devel_names = original_matched_devel_df.values[:,0]

        original_test_df = pd.read_csv(original_test_file, skiprows=6379, header=None)
        self.original_test_X = original_test_df.values[:, 1:label_index].astype(np.float32)
        self.original_test_y = original_test_df.values[:, label_index].astype(str)        
        self.original_test_names = original_test_df.values[:, 0]
        
        original_matched_test_df = pd.read_csv(original_matched_test_file, skiprows=6379, header=None)
        self.original_matched_test_X = original_matched_test_df.values[:, 1:label_index].astype(np.float32)
        self.original_matched_test_y = original_matched_test_df.values[:, label_index].astype(str)        
        self.original_matched_test_names = original_matched_test_df.values[:, 0]
        

        #naive splits
        naive_train_df = pd.read_csv(naive_train_file, skiprows=6379, header=None)
        self.naive_train_X = naive_train_df.values[:, 1:label_index].astype(np.float32)
        self.naive_train_y = naive_train_df.values[:, label_index].astype(str)
        self.naive_train_names = naive_train_df.values[:,0]
        
        naive_devel_df = pd.read_csv(naive_devel_file, skiprows=6379, header=None)
        self.naive_devel_X = naive_devel_df.values[:, 1:label_index].astype(np.float32)
        self.naive_devel_y = naive_devel_df.values[:, label_index].astype(str)
        self.naive_devel_names = naive_devel_df.values[:,0]
        
        naive_test_df = pd.read_csv(naive_test_file, skiprows=6379, header=None)
        self.naive_test_X = naive_test_df.values[:, 1:label_index].astype(np.float32)
        self.naive_test_y = naive_test_df.values[:, label_index].astype(str)        
        self.naive_test_names = naive_test_df.values[:, 0]

    def make_dict_json_serializable(self, meta_dict: dict) -> dict:
        '''
        for saving json files - cannot save np.ndarry
        '''
        cleaned_meta_dict = meta_dict.copy()
        for key in cleaned_meta_dict:
            if type(cleaned_meta_dict[key]) not in [str, float, int, np.float]:
                cleaned_meta_dict[key] = str(cleaned_meta_dict[key])
        return cleaned_meta_dict

    def bootstrap(self, best_estimator, X, y, test_X, test_y, random_state, train=False):
        '''
        function to perform bootstrapping for CIs currently not used but leaving in as maybe useful for
        other works
        inputs:
            best_estimator (SklearnModel): trained ML model
            X (ndarry): train features
            y (ndarry): train labels
            test_X (ndarry): test features
            test_y (ndarry): test labels
            random_state (int): random seed to ensure reproducibly randomness
            train (bool): do you also want to resample train?
        '''
        #if you want to boostrap on the training data
        if train:
            estimator = clone(best_estimator)
            sample_X, sample_y = resample(X, y, random_state=random_state)
            estimator.fit(sample_X, sample_y)
        
        #if you want to bookstrap on the test set 
        sample_test_X, sample_test_y = resample(test_X, test_y, random_state=random_state)
        auc_metrics = AUCSVM(best_estimator, sample_test_X, sample_test_y)
        _preds = best_estimator.predict(sample_test_X)

        _, prec, rec, pr_auc = auc_metrics.PR_AUC()
        _, fpr, tpr, roc_auc = auc_metrics.ROC_AUC()

        return recall_score(sample_test_y, _preds, average='macro'), prec, rec, pr_auc, fpr, tpr, roc_auc

          

    def run_svm(self,
            feature_folder,
            results_folder,
            params,
            train_scheme,
            train_X,
            train_y,
            train_names,
            devel_X,
            devel_y,
            devel_names,
            test_X,
            test_y,
            test_names,
            matched_test_X,
            matched_test_y,
            matched_test_names,
            long_test_X,
            long_test_y,
            long_test_names,
            long_matched_test_X,
            long_matched_test_y,
            long_matched_test_names
            ):
        '''
        perform the actual training and eval of the svm. 
        inputs:
            features folder (str): location of features
            results_folder (str): location of where to save the results
            params (dict): config of svm model
            .... (ndarry): all the test are np.arrays for training and eval
        '''

        train_X = np.nan_to_num(train_X)
        devel_X = np.nan_to_num(devel_X)

        self.metrics[train_scheme] = {}
        self.train_scheme = train_scheme
        num_train = train_X.shape[0]
        num_devel = devel_X.shape[0]
        num_test = test_X.shape[0]
        num_long_test = long_test_X.shape[0]
        num_matched_test = matched_test_X.shape[0]
        print(f'''num_train: {num_train},\
                num_devel: {num_devel},\
                num_test: {num_test},\
                num_long_test: {num_long_test},\
                num_matched_test: {num_matched_test}''')

        split_indices = np.repeat([-1, 0], [num_train, num_devel])
        split = PredefinedSplit(split_indices)
        X = np.append(train_X, devel_X, axis=0)
        y = np.append(train_y, devel_y, axis=0)
        grid_search = GridSearchCV(estimator=self.PIPELINE, param_grid=self.GRID, 
                                    scoring=make_scorer(recall_score, average='macro'), 
                                    n_jobs=-1, cv=split, refit=True, verbose=3, 
                                    return_train_score=False)
        
        # fit on data. train -> devel first, then train+devel implicit
        grid_search.fit(X, y)
        best_estimator = grid_search.best_estimator_
        
        # fit clone of best estimator on train again for devel predictions
        estimator = clone(best_estimator, safe=False)
        estimator.fit(train_X, train_y)
        preds = estimator.predict(devel_X)
        
        # devel metrics
        uar = recall_score(devel_y, preds, average='macro')
        cm = confusion_matrix(devel_y, preds)
        print(f'UAR: {uar}\n{classification_report(devel_y, preds)}\n\nConfusion Matrix:\n\n{cm}') 
        self.metrics[train_scheme]['dev'] = {}
        self.metrics[train_scheme]['dev']['uar'] = uar
        self.metrics[train_scheme]['dev']['cm'] = cm.tolist()
        self.metrics[train_scheme]['params'] = self.make_dict_json_serializable(grid_search.best_params_)

        df_predictions = pd.DataFrame({'filename': devel_names.tolist(), 'prediction': preds.tolist()})
        df_predictions.to_csv(os.path.join(results_folder, f'devel.predictions.csv'), index=False)

        pd.DataFrame(grid_search.cv_results_).to_csv(
            os.path.join(results_folder, f'grid_search.csv'), index=False)

        # test metrics
        print(f'Generating test predictions for optimised parameters {self.metrics[train_scheme]["params"]}')
        self.run_test(best_estimator, X, y, test_X, test_y, test_names, 'test', params, results_folder, train_scheme)
        if train_scheme != 'naive':
            self.run_test(best_estimator, X, y, long_test_X, long_test_y, long_test_names, 'long_test', params, results_folder, train_scheme)
            self.run_test(best_estimator, X, y, matched_test_X, matched_test_y, matched_test_names, 'matched_test', params, results_folder, train_scheme)
            self.run_test(best_estimator, X, y, long_matched_test_X, long_matched_test_y, long_matched_test_names, 'long_matched_test', params, results_folder, train_scheme)
        with open(os.path.join(results_folder, f'metrics.json'), 'w') as f:
            json.dump(self.metrics, f)

    def run_test(self, 
                estimator, 
                train_X, 
                train_y, 
                test_X:np.array, 
                test_y:np.array, 
                test_names, 
                test_name:str, 
                params, 
                results_folder,
                train_scheme):
        '''
        Run prediction and evaluation metrics for a test set: X, y
        inputs:
            estimator (SklearnClassifier): trained ML model
            please see run_svm for other input info
            train_scheme (str): how was the model trained? for logging

        '''
        print(test_name)
        test_X = np.nan_to_num(test_X)
        preds = estimator.predict(test_X)
        uar = recall_score(test_y, preds, average='macro')
        cm = confusion_matrix(test_y, preds)
        auc_metrics = AUCSVM(estimator, test_X, test_y)

        #plot_roc_curve(estimator, test_X, test_y, pos_label='Positive')
        fig, prec, rec, pr_auc = auc_metrics.PR_AUC()
        plt.title(f'{self.modality}_{test_name}')
        plt.savefig(f'{results_folder}{test_name}_PRcurve.png')
        plt.close()

        fig, fpr, tpr, roc_auc = auc_metrics.ROC_AUC()
        plt.plot([0,1], [0, 1], color='red', linestyle='--')
        plt.title(f'{self.modality}_{test_name}')
        plt.savefig(f'{results_folder}{test_name}_roc_curve.png')
        plt.close()

        self.metrics[train_scheme][test_name] = {}
        self.metrics[train_scheme][test_name]['precision'] = prec.tolist()
        self.metrics[train_scheme][test_name]['recall'] = rec.tolist()
        self.metrics[train_scheme][test_name]['fpr'] = fpr.tolist()
        self.metrics[train_scheme][test_name]['tpr'] = tpr.tolist()
        self.metrics[train_scheme][test_name]['pr_auc'] = pr_auc
        self.metrics[train_scheme][test_name]['roc_auc'] = roc_auc
        self.metrics[train_scheme][test_name]['uar'] = uar
        self.metrics[train_scheme][test_name]['cm'] = cm.tolist()
        print(f'UAR: {uar}\n{classification_report(test_y, preds)}\n\nConfusion Matrix:\n\n{cm}') 
        
        # No longer computing CIs through bootstrapping - using normal approximation method instead
        #print('Computing CI...')
        #bootstrap_results = Parallel(n_jobs=-1, verbose=10)(delayed(self.bootstrap)(estimator, train_X, train_y, test_X, test_y, i) for i in range(params['bootstrap_iterations']))
        #uars, prec, rec, pr_auc, fpr, tpr, roc_auc = zip(*bootstrap_results)
        #uar_ci_low, uar_ci_high = scipy.stats.t.interval(params['ci_interval'], len(uars)-1, loc=np.mean(uars), scale=scipy.stats.sem(uars))
        #pr_auc_ci_low, pr_auc_ci_high = scipy.stats.t.interval(params['ci_interval'], len(pr_auc)-1, loc=np.mean(pr_auc), scale=scipy.stats.sem(pr_auc))
        #roc_auc_ci_low, roc_auc_ci_high = scipy.stats.t.interval(params['ci_interval'], len(roc_auc)-1, loc=np.mean(roc_auc), scale=scipy.stats.sem(roc_auc))
        #self.metrics[test_name]['uar_ci_low'] = uar_ci_low
        #self.metrics[test_name]['uar_ci_high'] = uar_ci_high
        #self.metrics[test_name]['uar_mean'] = np.mean(uars)
        #self.metrics[test_name]['pr_auc_ci_low'] = pr_auc_ci_low
        #self.metrics[test_name]['pr_auc_ci_high'] = pr_auc_ci_high
        #self.metrics[test_name]['roc_auc_ci_low'] = roc_auc_ci_low
        #self.metrics[test_name]['roc_auc_ci_high'] = roc_auc_ci_high
        #self.metrics[test_name]['pr_auc_mean'] = np.mean(pr_auc)
        #self.metrics[test_name]['roc_auc_mean'] = np.mean(roc_auc)
        df_predictions = pd.DataFrame({'filename': test_names.tolist(), 'prediction': preds.tolist()})
        df_predictions.to_csv(os.path.join(results_folder, f'{test_name}.predictions.csv'), index=False)

    

if __name__=='__main__':
    params = {'ci_interval': 0.95, 'bootstrap_iterations': 1000}
    feature_base ='./features/opensmile_final/'
    for modality in POSSIBLE_MODALITIES:
        print('*'*20)
        print(f'Starting with modality: {modality}')
        print('*'*20)
        svm_instance = RunSvm(modality, feature_base)
        for train_scheme in ['standard', 'naive', 'matched']:#, 'original', 'original_matched']:
            result_base = f'./results_june_2022/{modality}/{train_scheme}/'
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
            elif train_scheme == 'original':
                train_X=svm_instance.original_train_X
                train_y=svm_instance.original_train_y
                train_y=svm_instance.original_train_names
                devel_X=svm_instance.original_devel_X
                devel_y=svm_instance.original_devel_y
                devel_names=svm_instance.original_devel_names
                test_X=svm_instance.original_test_X
                test_y=svm_instance.original_test_y
                test_names=svm_instance.original_test_names
                matched_test_X=svm_instance.original_matched_test_X
                matched_test_y=svm_instance.original_matched_test_y
                matched_test_names=svm_instance.original_matched_test_names
                long_test_X=svm_instance.long_test_X
                long_test_y=svm_instance.long_test_y
                long_test_names=svm_instance.long_test_names
                long_matched_test_X=svm_instance.long_matched_test_X
                long_matched_test_y=svm_instance.long_matched_test_y
                long_matched_test_names=svm_instance.long_matched_test_names
            elif train_scheme == 'original_matched':
                train_X=svm_instance.original_matched_train_X
                train_y=svm_instance.original_matched_train_y
                train_names=svm_instance.original_matched_train_names
                devel_X=svm_instance.original_devel_X
                devel_y=svm_instance.original_devel_y
                devel_names=svm_instance.original_devel_names
                test_X=svm_instance.original_test_X
                test_y=svm_instance.original_test_y
                test_names=svm_instance.original_names
                matched_test_X=svm_instance.original_matched_test_X
                matched_test_y=svm_instance.original_matched_test_y
                matched_test_names=svm_instance.original_matched_test_names
                long_test_X=svm_instance.long_test_X
                long_test_y=svm_instance.long_test_y
                long_test_names=svm_instance.long_test_names
                long_matched_test_X=svm_instance.long_matched_test_X
                long_matched_test_y=svm_instance.long_matched_test_y
                long_mathced_test_names=svm_instance.long_matched_test_names
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
