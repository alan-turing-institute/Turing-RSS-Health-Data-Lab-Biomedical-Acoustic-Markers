import os
import json
import re
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

from svm import RunSvm
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from utils.eval_metrics import AUCSVM

class WeakSvm(RunSvm):

    '''
    for now I am just running on the naive set as proof of concept
    '''
    def __init__(self, modality, feature_base, results_base):
        super(WeakSvm, self).__init__(modality, feature_base)
        self.results_base = results_base + modality
        self.modality = modality
        if not os.path.exists(self.results_base):
            os.makedirs(self.results_base)
        self.metrics = {}
        self.pca_explained_variance_ratio = {}
        #naive exp
        self.pca_test_y = self.naive_test_y
        self.pca_test_names = self.naive_test_names
        self.pca_test_X = self.naive_test_X
        #we are not performing cross validation he
        self.metrics['naive'] = {}
        self.pca_train_y = self.concat(self.naive_train_y, self.naive_devel_y)
        self.pca_train_names = self.concat(self.naive_train_names, self.naive_devel_names)
        self.pca_train_X = self.concat(self.naive_train_X, self.naive_devel_X)
        print('starting pca extraction')
        self.pca_fit('naive', covid_neg=True)
        self.iterative_less_weak('naive')
        #match exp
        self.metrics['matched'] = {}
        self.pca_test_y = self.matched_test_y
        self.pca_test_names = self.matched_test_names
        self.pca_test_X = self.matched_test_X
        #we are not performing cross validation he
        self.pca_train_y = self.concat(self.matched_train_y, self.matched_devel_y)
        self.pca_train_names = self.concat(self.matched_train_names, self.matched_devel_names)
        self.pca_train_X = self.concat(self.matched_train_X, self.matched_devel_X)

        print('starting pca extraction')
        self.pca_fit('matched', covid_neg=True)
        self.iterative_less_weak('matched')
        
        with open(os.path.join(self.results_base, f'metrics.json'), 'w') as f:
            json.dump(self.metrics, f)


    def concat(self, train, devel):
        return np.append(train, devel, axis=0)

    def pca_fit(self, split, covid_neg=False):
        '''
        perform pca compression
        if covid_neg == True create the components based solely on the COVID neg
        cases
        TODO: fit pca to just covid neg variance
        '''
        self.pca_components = min(200, len(self.train_names)-1)
        pca = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=self.pca_components))])
        if covid_neg == True:
            covid_neg_train = np.where(self.pca_test_y == 'Negative')[0]
            covid_neg_train = self.pca_train_X[covid_neg_train]
            print(len(self.pca_train_names))
            print(len(covid_neg_train))

            pca.fit(np.nan_to_num(covid_neg_train))
        else:
            pca.fit(np.nan_to_num(self.pca_train_X))
        self.pca_train_X = pca.transform(np.nan_to_num(self.pca_train_X))
        self.pca_test_X = pca.transform(np.nan_to_num(self.pca_test_X))
        self.pca_explained_variance_ratio[split] = pca['pca'].explained_variance_ratio_

    def iterative_less_weak(self, scheme):
        '''
        train and evaluate an svm model which has iteratively more pca components
        '''
        for num_components in range(1, self.pca_components+1):
            print(f'Running SVM with {num_components} number of pca components')
            train_X = self.pca_train_X[:,:num_components]
            test_X = self.pca_test_X[:,:num_components]
            svm = make_pipeline(
                    StandardScaler(),
                    LinearSVC(
                        random_state=42,
                        class_weight='balanced',
                        max_iter=10000,
                        loss='squared_hinge',
                        C=0.0001))
            svm.fit(train_X, self.pca_train_y) 
            preds = svm.predict(test_X)
            self.run_test(
                    svm,
                    train_X,
                    self.pca_train_y,
                    test_X,
                    self.pca_test_y,
                    self.pca_test_names,
                    f'{num_components}_components',
                    None,
                    self.results_base,
                    scheme)


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
            '''
            test_X = np.nan_to_num(test_X)
            preds = estimator.predict(test_X)
            uar = recall_score(test_y, preds, average='macro')
            cm = confusion_matrix(test_y, preds)
            auc_metrics = AUCSVM(estimator, test_X, test_y)

            #plot_roc_curve(estimator, test_X, test_y, pos_label='Positive')
            fig, prec, rec, pr_auc = auc_metrics.PR_AUC()
            fig, fpr, tpr, roc_auc = auc_metrics.ROC_AUC()

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

            df_predictions = pd.DataFrame({'filename': test_names.tolist(), 'prediction': preds.tolist()})
            df_predictions.to_csv(os.path.join(results_folder, f'{test_name}.predictions.csv'), index=False)

class MNISTWeakSVM(WeakSvm):
    def __init__(self, modality, feature_base, results_base):
        self.modality = modality
        label_index = -1
        self.feature_base = feature_base
        self.results_base = results_base + modality
        self.pca_explained_variance_ratio = {}
        if not os.path.exists(self.results_base):
            os.makedirs(self.results_base)
        self.metrics = {}
        self.metrics[self.modality] = {}
        train_file = os.path.join(feature_base,
                                     'train.csv')
        train_df = pd.read_csv(train_file, skiprows=6379, header=None)
        self.train_X = train_df.values[:, 1:label_index].astype(np.float32)
        self.train_y = train_df.values[:, label_index].astype(str)
        self.train_names = train_df.values[:,0]

        test_file = os.path.join(feature_base,
                                     'test.csv')
        test_df = pd.read_csv(test_file, skiprows=6379, header=None)
        self.test_X = test_df.values[:, 1:label_index].astype(np.float32)
        self.test_y = test_df.values[:, label_index].astype(str)
        self.test_names = test_df.values[:,0]

        self.pca_test_y = self.test_y
        self.pca_test_names = self.test_names
        self.pca_test_X = self.test_X
        #we are not performing cross validation he
        self.pca_train_y = self.train_y
        self.pca_train_names = self.train_names, 
        self.pca_train_X = self.train_X
        assert not any([x in self.pca_test_names for x in self.pca_train_names]), 'There appears to be a cross over between train and test'
        print('starting pca extraction')
        self.pca_fit('standard')
        self.iterative_less_weak(self.modality)

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
        '''
        test_X = np.nan_to_num(test_X)
        preds = estimator.predict(test_X)
        uar = recall_score(test_y, preds, average='macro')
        cm = confusion_matrix(test_y, preds)
        #if self.modality == 'mnist':
        #    auc_metrics = MultiAUCSVM(estimator, test_X, test_y)
        #    fpr, tpr, roc_auc = auc_metrics.ROC_AUC()
        #else:
        #    auc_metrics = AUCSVM(estimator, test_X, test_y)
        #    fig, fpr, tpr, roc_auc = auc_metrics.ROC_AUC()
        #print(roc_auc)

        self.metrics[train_scheme][test_name] = {}
        #self.metrics[train_scheme][test_name]['pr_auc'] = np.mean(pr_auc)
        #self.metrics[train_scheme][test_name]['roc_auc'] = np.mean(roc_auc)
        self.metrics[train_scheme][test_name]['uar'] = uar
        self.metrics[train_scheme][test_name]['cm'] = cm.tolist()
        print(f'UAR: {uar}\n{classification_report(test_y, preds)}\n\nConfusion Matrix:\n\n{cm}')

        df_predictions = pd.DataFrame({'filename': test_names.tolist(), 'prediction': preds.tolist()})
        df_predictions.to_csv(os.path.join(results_folder, f'{test_name}.predictions.csv'), index=False)


class MultiAUCSVM(AUCSVM):
    '''
    same as aucsvm but with multiclass!
    '''
    def __init__(self, estimator, X, y):
        super(MultiAUCSVM, self).__init__(estimator, X, y)
    
    def PR_AUC(self, **kwargs):
        prec, rec, area = [], [], []
        for i in range(np.shape(self.y_preds)[1]):
            p, r, _ = precision_recall_curve(self.y, self.y_preds[:,i], pos_label=str(i))
            a = average_precision_score(self.y, self.y_preds[:,i], pos_label=str(i))
            prec.append(p)
            rec.append(r)
            area.append(a)
        return prec, rec, area

    def ROC_AUC(self, **kwargs):
        fpr, tpr, area = [], [], []
        print(self.y_preds)
        print(self.y)
        for i in range(np.shape(self.y_preds)[1]):

            f, t, _ = roc_curve(self.y, self.y_preds[:,i], pos_label=str(i))
            a = auc(f, t)
            fpr.append(f)
            tpr.append(t)
            area.append(a)
        return fpr, tpr, area

def plot_weak_curves(metrics1, metrics2, fig_base):
    '''
    plot how the metrics improve with the number of pca components used for svm
    '''
    naive = []
    matched = []
    ciab_explained_naive = []
    ciab_explained_cum_naive = 0
    ciab_explained_match = []
    ciab_explained_cum_match = 0
    mnist_explained = []
    mnist_explained_cum = 0
    mnist = []
    for num_components in range(1, metrics2.pca_components+1):
        naive.append(metrics2.metrics['naive'][f'{num_components}_components']['uar'])
        matched.append(metrics2.metrics['matched'][f'{num_components}_components']['uar'])
        ciab_explained_cum_naive += metrics2.pca_explained_variance_ratio['naive'][num_components-1]
        ciab_explained_naive.append(ciab_explained_cum_naive)
        ciab_explained_cum_match += metrics2.pca_explained_variance_ratio['matched'][num_components-1]
        ciab_explained_match.append(ciab_explained_cum_match)
    for num_components in range(1, metrics1.pca_components+1):
        mnist.append(metrics1.metrics['esc50'][f'{num_components}_components']['uar'])
        mnist_explained_cum += metrics1.pca_explained_variance_ratio['standard'][num_components-1]
        mnist_explained.append(mnist_explained_cum)
    return naive, matched, mnist, ciab_explained_naive, ciab_explained_match, mnist_explained


def robust_ssast(target, ssast_predicts, weak_base):
    '''
    Given the performance of the weak model iteratively cull the easy to classify examples from the 
    test set as these are hypothesised to contain identifiable bias
    '''
    ssast = pd.read_csv(ssast_predicts, names=['Negative', 'Positive'])
    labels = pd.read_csv(target, names=['Negative', 'Positive'])
    ssast['pred'] = ssast.apply(lambda x: 'Negative' if x['Negative'] > x['Positive'] else 'Positive', axis=1)
    ssast['labels'] = labels.apply(lambda x: 'Negative' if x['Negative'] > x['Positive'] else 'Positive', axis=1)
    ssast = map_idex_to_id(ssast, 'ssast_results/ciab_matched_test_data_1.json')
    ssast_score, upper, lower = [], [], []
    random_score, random_upper, random_lower = [], [], []
    uar, u, l = calc_recall(ssast)
    ssast_score.append(uar)
    upper.append(u)
    lower.append(l)
    random_score.append(uar)
    random_upper.append(u)
    random_lower.append(l)
    for num_components in range(1, 201):
        weak_preds = pd.read_csv(os.path.join(weak_base, f'{num_components}_components.predictions.csv'))
        weak_preds['filename'] = weak_preds['filename'].apply(lambda x: x.replace("'", ""))
        weak_preds = weak_preds.merge(ssast, left_on='filename', right_on='id')
        weak_ssast_preds = weak_preds[weak_preds['prediction'] != weak_preds['labels']] #remove all cases where the weak model gets the prediction correct

        random_preds = random_removal(
                weak_preds, 
                len(weak_preds[weak_preds['prediction'] != weak_preds['labels']]))

        uar, u, l = calc_recall(weak_ssast_preds)
        ssast_score.append(uar)
        upper.append(u)
        lower.append(l)
        uar, u, l = calc_recall(random_preds)
        random_score.append(uar)
        random_upper.append(u)
        random_lower.append(l)
    return ssast_score, upper, lower, random_score, random_upper, random_lower   

def random_removal(df, num_to_remove):
    '''
    randomly remove cases from the matched test set - serves as a comparision
    '''
    random_index = np.random.choice(df.index, num_to_remove, replace=False)
    drop_df = df.drop(random_index)

    left = len(df) - num_to_remove
    assert len(drop_df) == left, f'This should total the same. {left} != {len(drop_df)}'
    return drop_df
        

def map_idex_to_id(df, mapping_path):

    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    df['id'] = df.apply(lambda x: re.findall(r'[^\/]+$', mapping['data'][x.name]['wav'])[0], axis=1)
    return df
    

def calc_recall(df):
    uar =  recall_score(df['labels'], df['pred'], average='macro')
    conf = 1.960 * ((uar*(1 - uar))/(len(df[df['labels'] == 'Positive']) + len(df[df['labels'] == 'Negative'])))**0.5
    return uar, uar + conf, uar - conf

def find_thres(x_point, x, y):
    for x_i, y_i in zip(x,y):
        if x_i >= x_point:
            return y_i
    raise ValueError('x does not cross the thresehold!')
    
if __name__ == '__main__':
    feature_base ='./features/opensmile_final/'
    results_base= './results_july_2022/'
    mnist_instance = MNISTWeakSVM('esc50', './features/opensmile_esc50/', results_base)
    svm_instance = WeakSvm('audio_sentence_url', feature_base, results_base)
    naive, matched, mnist, ciab_explained_naive, ciab_explained_match, cough_explained = plot_weak_curves(mnist_instance, svm_instance, results_base)
    ssast_score, upper, lower, random_score, random_upper, random_lower = robust_ssast('ssast_results/target.csv', 'ssast_results/predictions_matched_set.csv', 'results_july_2022/audio_sentence_url')

    fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()

    #ax2.plot(cough_explained, mnist, label='weak-model-coughvslaugh', color='tab:blue')
    #ax2.plot(ciab_explained_naive, naive, label='weak-model-covid-naive', color='tab:blue', linestyle='dashed')
    ax1.plot(ciab_explained_match, matched, label='weak-model-covid-matched', color='tab:blue')#, linestyle='dashdot')
    ax1.plot([0] + ciab_explained_match, ssast_score, label='ssast-covid-matched-curated-removal', color='tab:red')
    ax1.plot([0] + ciab_explained_match, random_score, label='ssast-covid-matched-random-removal', color='tab:purple')#, linestyle='dashed')
    ax1.fill_between([0] + ciab_explained_match, upper, lower, alpha=0.3, color='tab:red')
    ax1.fill_between([0] + ciab_explained_match, random_upper, random_lower, alpha=0.3, color='tab:purple')
    ax1.set_xlabel('Proportion of variance explained by cumulative PCA components')
    ax1.set_ylabel('UAR')#, color='tab:red')
    ax1.plot([0.5, 0.5], [0.35, 0.65], color='tab:green', label='calibration-threshold')
    y_thres = find_thres(0.5,[0] + ciab_explained_match, ssast_score) 
    ax1.fill_between(
            [0] + ciab_explained_match,
            ssast_score,
            [y_thres]*len([0]+ciab_explained_match),
            where=[i>j for i, j in zip(ssast_score,[y_thres]*len([0]+ciab_explained_match))], alpha=0.3, color='tab:green', interpolate=True, label='performance-due-to-confouding')
    #ax1.yaxis.label.set_color('tab:red')
    #ax1.spines['left'].set_color('tab:red')
    #ax1.tick_params(axis='y', colors='tab:red')
    #ax2.set_ylabel('UAR-weak', color='tab:blue')
    #ax2.yaxis.label.set_color('tab:blue')
    #ax2.spines['right'].set_color('tab:blue')
    #ax2.spines['left'].set_color('tab:red')
    #ax2.tick_params(axis='y', colors='tab:blue')
    fig.legend(loc="upper right", bbox_to_anchor=(1.7,1), bbox_transform=ax1.transAxes)
    plt.savefig(f'results_july_2022/weak_robust8.png', bbox_inches='tight')
