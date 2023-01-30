
'''
Script to estimate the bounds of the bayes error and twice the bayes error
author: Harry Coppock
Qs: harry.coppock@imperial.ac.uk
'''

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import scipy
from glob import glob
import os
import re
import pandas as pd
import seaborn as sns
import boto3
import io
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
import sys
sys.path.append('../')
from ssast.src.finetune.ciab.prep_ciab import PrepCIAB


def load_features(path):
    cols = list(pd.read_csv(path, nrows =1))
    df= pd.read_csv(path, usecols =[i for i in cols if i != 'Unnamed: 0'])
    df = reformat_names(df)
    list_features= list(df.columns)
    list_features.remove('barcode')
    #load the meta data
    meta_data = load_meta()
    df.rename(columns={'barcode':'audio_sentence'}, inplace=True)
    data_df = merge_features_meta(df, meta_data)
    data_df['coughq'] = control_cough(data_df)
    data_df['recruitment_source'] = control_recruitment_source(data_df)
    data_df['age'] = data_df['age'].apply(lambda x: float(x))
    return data_df, list_features

def reformat_names(df):
    df['barcode'] = df['barcode'].apply(lambda x: x.rsplit('/', 1)[-1])
    return df


def merge_features_meta(features, meta_data):
    return features.merge(meta_data,
                            how='inner',
                             on='audio_sentence')

def get_file(path, bucket):
    return io.BytesIO(bucket.Object(path).get()['Body'].read())

def load_meta():
    extract = PrepCIAB()
    return extract.meta_data

def control_cough(df):
    selection = df.symptoms.apply(
                            lambda x: any(symptom == 'Cough (any)' for symptom in x))
    return selection

def control_recruitment_source(df):
    selection = df.recruitment_source.apply(
                            lambda x:'Test and Trace' if 'REACT' not in x else 'REACT') 
    return selection

def KNN(df_train, df_test, label, list_features):
    '''
    Perform 1 nearest neighbor fitting for the specified label e.g. test result/symptom
    now generating bootstrapped results so need to resample from train and test
    df_train: pandas dataframe for training
    df_tets: pandas dataframe for testing
    label: str corresponding to a column in df
    list_features: list of column names corresponding to the learn feature dims
    '''
    train = df_train.dropna(subset=[label])
    test= df_test.dropna(subset=[label])
    #resample with replacement new datasets
    test = resample(test)
    train = resample(train)
    #train a KNN algo on resampled train and eval on resampled test
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(train[list_features], train[label])

    preds = neigh.predict(test[list_features])
    acc = accuracy_score(test[label], preds)
    uar = recall_score(test[label], preds, average='macro')
    return uar 


def plot_summary_bar(metrics, order, modality):
    x = np.arange(len(order))  # the label locations
    width = 0.15  # the width of the bars
    fig, ax = plt.subplots(figsize=(10,8))
    rects1 = ax.bar(x - 2*width,
            metrics['test_result_mean'],
            width,
            label='test_result',
            color='#007C91',
            capsize=3,
            yerr=[np.array(metrics['test_result_mean']) - np.array(metrics['test_result_low']), np.array(metrics['test_result_high']) - np.array(metrics['test_result_mean'])])
    rects2 = ax.bar(x - width,
            metrics['coughq_mean'],
            width,
            label='coughq',
            color='#003B5C',
            capsize=3,
            yerr=[np.array(metrics['coughq_mean']) - np.array(metrics['coughq_low']), np.array(metrics['coughq_high']) - np.array(metrics['coughq_mean'])])
    rects3 = ax.bar(x,
            metrics['gender_mean'],
            width,
            label='gender',
            color='#582C83',
            capsize=3,
            yerr=[np.array(metrics['gender_mean']) - np.array(metrics['gender_low']), np.array(metrics['gender_high']) - np.array(metrics['gender_mean'])])
    rects4 = ax.bar(x + width,
            metrics['viral_load_cat_mean'],
            width,
            label='viral_load_cat',
            color='#8A1B61',
            capsize=3,
            yerr=[np.array(metrics['viral_load_cat_mean']) - np.array(metrics['viral_load_cat_low']), np.array(metrics['viral_load_cat_high']) - np.array(metrics['viral_load_cat_mean'])])
    rects5 = ax.bar(x + 2*width,
            metrics['recruitment_source_mean'],
            width,
            label='recruitment_source',
            color='#E40046',
            capsize=3,
            yerr=[np.array(metrics['recruitment_source_mean']) - np.array(metrics['recruitment_source_low']), np.array(metrics['recruitment_source_high']) - np.array(metrics['recruitment_source_mean'])])


    #  Add some text for labels, title and custom x-axis tick labels, etc.
    #ax.set_ylim(bottom=0.3)
    ax.set_ylabel('UAR')
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.tick_params(axis='x', rotation=60)
    ax.legend()
    ax.grid(True)
    plt.savefig(f'figs/ssast/bayes_error_{modality}.png',
    bbox_inches='tight'
    )



if __name__ == '__main__':
    POSSIBLE_train = ['standard-train', 'naive', 'big', 'matched-train']
    POSSIBLE_test = ['standard_test', 'matched_test', 'long_test']#, 'analysis_train']
    POSSIBLE_MODALITIES = [#'audio_cough_url',
                        #'three_cough',
                        'sentence']
                        #'audio_ha_sound_url']

    POSSIBLE_CONTROLS = [
                #"age",
                "coughq",
                "gender",
                "viral_load_cat",
                #"smoker_status",
                "recruitment_source",
                #"language",
                "test_result"]

    POSSIBLE_METHODS =[
                "tsne"]#,
                #"pca"]
    metrics = {}
    order = []
    for train_method in tqdm(POSSIBLE_train):
        for row, test_method in enumerate(POSSIBLE_test):
            for method in POSSIBLE_METHODS:
                for modality in POSSIBLE_MODALITIES:
                    if train_method == 'naive' and (test_method == 'matched_test' or test_method == 'long_test'):
                        continue
                    if train_method == 'big' and test_method == 'long_test':
                        continue
                    test_path = f'/home/ec2-user/SageMaker/jbc-cough-in-a-box/ssast/src/finetune/ciab/exp/test01-ciab_{modality}-f128-128-t1-2-b20-lr1e-4-ft_avgtok-base-unknown-SSAST-Base-Frame-400-1x-noiseTrue-{train_method}-final/fold1/{test_method}_pca_projections.csv'
                    train_path = f'/home/ec2-user/SageMaker/jbc-cough-in-a-box/ssast/src/finetune/ciab/exp/test01-ciab_{modality}-f128-128-t1-2-b20-lr1e-4-ft_avgtok-base-unknown-SSAST-Base-Frame-400-1x-noiseTrue-{train_method}-final/fold1/analysis_train_pca_projections.csv'
                    if not os.path.exists(f'.figs/ssast/{train_method}/{test_method}'):
                        os.makedirs(f'.figs/ssast/{train_method}/{test_method}')
                    data_test_df, list_features_test = load_features(test_path)
                    data_train_df, list_features_train = load_features(train_path)
                    assert list_features_test == list_features_train, 'bug in feature dims'
                    
                    for column, control in enumerate(POSSIBLE_CONTROLS):
                        if f'{control}_mean' not in metrics.keys():
                            metrics[f'{control}_mean'] = []
                            metrics[f'{control}_low'] = []
                            metrics[f'{control}_high'] = []
                        if control == 'test_result':
                            if train_method == 'naive':
                                order.append('naive')
                            else:
                                order.append(f'{train_method} \n {test_method}')
                        bootstrap_results = Parallel(n_jobs=-1, verbose=10)(delayed(KNN)(data_train_df, data_test_df, control, list_features_test) for i in range(10))
                        acc = bootstrap_results
                        acc_ci_low, acc_ci_high = scipy.stats.t.interval(0.95, len(acc)-1, loc=np.mean(acc), scale=scipy.stats.sem(acc))
                        metrics[f'{control}_mean'].append(np.mean(acc))
                        metrics[f'{control}_low'].append(acc_ci_low)
                        metrics[f'{control}_high'].append(acc_ci_high)

    plot_summary_bar(metrics, order, modality)
