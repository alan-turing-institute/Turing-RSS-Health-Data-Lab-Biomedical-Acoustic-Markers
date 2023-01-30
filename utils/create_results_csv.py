'''
simple script to collect all the results and save them as a csv for easy latex table creation
'''
import pandas as pd
import json 
import os
import sys

from eval_metrics import EvalMetrics
from dataset_stats import DatasetStats

def return_svm_results(results):
    svm_results_path = '../SvmBaseline/results_june_2022'
    results['SVM'] = {}
    for modality in os.listdir(svm_results_path):
        print(f'Modality: {modality}')
        metric_path = os.path.join(svm_results_path, modality, 'matched/metrics.json')
        with open(metric_path) as f:
            metrics = json.load(f)

        results['SVM'][modality] = {}

        s_t = metrics['standard']
        m_t = metrics['matched']
        n_t = metrics['naive']
        results['SVM'][modality]['UAR'] = [s_t['test']['uar'],
                                    s_t['matched_test']['uar'],
                                    s_t['long_test']['uar'],
                                    s_t['long_matched_test']['uar'],
                                    m_t['test']['uar'],
                                    m_t['matched_test']['uar'],
                                    m_t['long_test']['uar'],
                                    m_t['long_matched_test']['uar'],
                                    n_t['test']['uar']
                                    ]
        results['SVM'][modality]['ROC'] = [s_t['test']['roc_auc'],
                                    s_t['matched_test']['roc_auc'],
                                    s_t['long_test']['roc_auc'],
                                    s_t['long_matched_test']['roc_auc'],
                                    m_t['test']['roc_auc'],
                                    m_t['matched_test']['roc_auc'],
                                    m_t['long_test']['roc_auc'],
                                    m_t['long_matched_test']['roc_auc'],
                                    n_t['test']['roc_auc']
                                    ]
        results['SVM'][modality]['PR'] = [s_t['test']['pr_auc'],
                                    s_t['matched_test']['pr_auc'],
                                    s_t['long_test']['pr_auc'],
                                    s_t['long_matched_test']['pr_auc'],
                                    m_t['test']['pr_auc'],
                                    m_t['matched_test']['pr_auc'],
                                    m_t['long_test']['pr_auc'],
                                    m_t['long_matched_test']['pr_auc'],
                                    n_t['test']['pr_auc']
                                    ]
    return results

def get_mod_ssast(name):
    if 'three_cough' in name:
        return 'Three cough'
    elif 'cough' in name:
        return 'Cough'
    elif 'sentence' in name:
        return 'Sentence'
    elif 'ha_sound' in name:
        return 'Ha sound'
    else:
        raise 'This should not happen'
def get_exp_ssast(name):
    if 'standard-train' in name:
        return 'Standard'
    elif 'matched-train' in name:
        return 'Match'
    elif 'naive' in name:
        return 'Naive'
    else:
        raise 'This should not happen'

def return_ssast_results(results):

    results['SSAST'] = {}
    ssast_results_path = '../ssast_ciab/src/finetune/ciab/exp/final'
    for experiment in os.listdir(ssast_results_path):
        print(f'Experiment: {experiment}')
        if 'original' in experiment:
            continue
        modality = get_mod_ssast(experiment)
        train = get_exp_ssast(experiment)
        metric_path = os.path.join(ssast_results_path, experiment, 'fold1/metrics.json')

        with open(metric_path) as f:
            metrics = json.load(f)
        if modality not in results['SSAST'].keys():
            results['SSAST'][modality] = {}
            results['SSAST'][modality]['UAR'] = {}
            results['SSAST'][modality]['ROC'] = {}
            results['SSAST'][modality]['PR'] = {}
        if train == 'Naive':

            results['SSAST'][modality]['UAR'][train] = [metrics['test']['uar'], ]
            results['SSAST'][modality]['ROC'][train] = [metrics['test']['auc']]
            results['SSAST'][modality]['PR'][train] = [metrics['test']['AP']]

        else:

            results['SSAST'][modality]['UAR'][train] = [metrics['test']['uar'],
                                                        metrics['matched_test']['uar'],
                                                        metrics['long_test']['uar'],
                                                        metrics['matched_long_test']['uar']]
            results['SSAST'][modality]['ROC'][train] = [metrics['test']['auc'],
                                                        metrics['matched_test']['auc'],
                                                        metrics['long_test']['auc'],
                                                        metrics['matched_long_test']['auc']]
            results['SSAST'][modality]['PR'][train] = [metrics['test']['AP'],
                                                        metrics['matched_test']['AP'],
                                                        metrics['long_test']['AP'],
                                                        metrics['matched_long_test']['AP']]

    results['SSAST']['Sentence']['UAR'] = results['SSAST']['Sentence']['UAR']['Standard'] + results['SSAST']['Sentence']['UAR']['Match'] +results['SSAST']['Sentence']['UAR']['Naive']
    results['SSAST']['Sentence']['ROC'] = results['SSAST']['Sentence']['ROC']['Standard'] + results['SSAST']['Sentence']['ROC']['Match'] +results['SSAST']['Sentence']['ROC']['Naive']
    results['SSAST']['Sentence']['PR'] = results['SSAST']['Sentence']['PR']['Standard'] + results['SSAST']['Sentence']['PR']['Match'] +results['SSAST']['Sentence']['PR']['Naive']
    
    results['SSAST']['Three cough']['UAR'] = results['SSAST']['Three cough']['UAR']['Standard'] + results['SSAST']['Three cough']['UAR']['Match'] +results['SSAST']['Three cough']['UAR']['Naive']
    results['SSAST']['Three cough']['ROC'] = results['SSAST']['Three cough']['ROC']['Standard'] + results['SSAST']['Three cough']['ROC']['Match'] +results['SSAST']['Three cough']['ROC']['Naive']
    results['SSAST']['Three cough']['PR'] = results['SSAST']['Three cough']['PR']['Standard'] + results['SSAST']['Three cough']['PR']['Match'] +results['SSAST']['Three cough']['PR']['Naive']
    
    results['SSAST']['Cough']['UAR'] = results['SSAST']['Cough']['UAR']['Standard'] + results['SSAST']['Cough']['UAR']['Match'] +results['SSAST']['Cough']['UAR']['Naive']
    results['SSAST']['Cough']['ROC'] = results['SSAST']['Cough']['ROC']['Standard'] + results['SSAST']['Cough']['ROC']['Match'] +results['SSAST']['Cough']['ROC']['Naive']
    results['SSAST']['Cough']['PR'] = results['SSAST']['Cough']['PR']['Standard'] + results['SSAST']['Cough']['PR']['Match'] +results['SSAST']['Cough']['PR']['Naive']

    results['SSAST']['Ha sound']['UAR'] = results['SSAST']['Ha sound']['UAR']['Standard'] + results['SSAST']['Ha sound']['UAR']['Match'] +results['SSAST']['Ha sound']['UAR']['Naive']
    results['SSAST']['Ha sound']['ROC'] = results['SSAST']['Ha sound']['ROC']['Standard'] + results['SSAST']['Ha sound']['ROC']['Match'] +results['SSAST']['Ha sound']['ROC']['Naive']
    results['SSAST']['Ha sound']['PR'] = results['SSAST']['Ha sound']['PR']['Standard'] + results['SSAST']['Ha sound']['PR']['Match'] +results['SSAST']['Ha sound']['PR']['Naive']
    return results


def convert_to_pd(results):

    index_names = [
            ('Sentence', 'SVM', 'UAR'),
            ('Sentence', 'SVM', 'ROC'),
            ('Sentence', 'SVM', 'PR'),
            ('Sentence', 'SSAST', 'UAR'),
            ('Sentence', 'SSAST', 'ROC'),
            ('Sentence', 'SSAST', 'PR'),
            ('Three cough', 'SVM', 'UAR'),
            ('Three cough', 'SVM', 'ROC'),
            ('Three cough', 'SVM', 'PR'),
            ('Three cough', 'SSAST', 'UAR'),
            ('Three cough', 'SSAST', 'ROC'),
            ('Three cough', 'SSAST', 'PR'),
            ('Cough', 'SVM', 'UAR'),
            ('Cough', 'SVM', 'ROC'),
            ('Cough', 'SVM', 'PR'),
            ('Cough', 'SSAST', 'UAR'),
            ('Cough', 'SSAST', 'ROC'),
            ('Cough', 'SSAST', 'PR'),
            ('Ha sound', 'SVM', 'UAR'),
            ('Ha sound', 'SVM', 'ROC'),
            ('Ha sound', 'SVM', 'PR'),
            ('Ha sound', 'SSAST', 'UAR'),
            ('Ha sound', 'SSAST', 'ROC'),
            ('Ha sound', 'SSAST', 'PR'),
            ]
    column_names = [
            ('Standard', 'Standard'),
            ('Standard', 'Match'),
            ('Standard', 'Long'),
            ('Standard', 'Long Matched'),
            ('Match', 'Standard'),
            ('Match', 'Match'),
            ('Match', 'Long'),
            ('Match', 'Long Matched'),
            ('Naive', 'Naive')
            ]
    print(results['SVM'].keys())
    results['SVM']['Sentence'] = results['SVM'].pop('audio_sentence_url')
    results['SVM']['Three cough'] = results['SVM'].pop('audio_three_cough_url')
    results['SVM']['Cough'] = results['SVM'].pop('audio_cough_url')
    results['SVM']['Ha sound'] = results['SVM'].pop('audio_ha_sound_url')
    data = []
    for (modality, model, metric) in index_names:
        if model not in results.keys():
            data.append([0,0,0,0,0,0,0,0,0])
            continue
        data.append(results[model][modality][metric])
    full_results= {
            'index': index_names,
            'columns': column_names,
            'data': data,
            'index_names': ['mode', 'model', 'metric'],
            'column_names': ['train', 'test']} 
    print(data)
    try:
        df = pd.DataFrame.from_dict(full_results, orient='tight')
    except ValueError:
        raise ValueError('The currnet Pytorch image (used as the base image for this exp docker uses python 3.7. Orient=tight requires pandas 1.4.0 which requires python 3.8. To use this script please simply run outside of docker container. see this issue for more details and updates https://github.com/pytorch/pytorch/issues/73714')
    df = df.round(decimals=3)
    print(df)
    sys.exit()

    df.to_csv('results.csv')

def add_conf():
    pass



if __name__ == '__main__':

    conf_int = EvalMetrics.conf_int
    dataset = DatasetStats()
    results = {}
    results = return_svm_results(results)
    results = return_ssast_results(results)
    convert_to_pd(results)



