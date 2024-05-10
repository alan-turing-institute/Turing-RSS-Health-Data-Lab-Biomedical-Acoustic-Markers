'''
Script to plot timeline of the submission dates of the train, test, long_test and matched test
author: Harry Coppock
qs to: harry.coppock@imperial.ac.uk
'''
import pandas as pd
import sys
from itertools import chain
import matplotlib.pyplot as plt
sys.path.append('../')
from SvmBaseline.opensmile_feat_extraction import ExtractOpensmile 
from utils.dataset_stats import DatasetStats

def sub_time(e):
    train_df = e.meta_data[e.meta_data['audio_sentence'].apply(lambda x: x in e.train)]
    test_df = e.meta_data[e.meta_data['audio_sentence'].apply(lambda x: x in e.test)]
    long_test_df = e.meta_data[e.meta_data['audio_sentence'].apply(lambda x: x in e.long_test.values.tolist())]
    matched_test_df = e.meta_data[e.meta_data['audio_sentence'].apply(lambda x: x in e.matched_test.values.tolist())]
    left_out_df = e.meta_data[e.meta_data['audio_sentence'].apply(lambda x: x not in e.matched_test.values.tolist() + e.long_test.values.tolist()+ e.train + e.test)]
    train_df = train_df.sort_values('submission_time', ascending=True)
    test_df = test_df.sort_values('submission_time', ascending=True)
    long_test_df = long_test_df.sort_values('submission_time', ascending=True)
    matched_test_df = matched_test_df.sort_values('submission_time', ascending=True)
    left_out_df = left_out_df.sort_values('submission_time', ascending=True)
    train_df['count'] = 1
    test_df['count'] = 1
    long_test_df['count'] = 1
    matched_test_df['count'] = 1
    left_out_df['count'] = 1
    train_df['cumulative'] = train_df['count'].cumsum()
    test_df['cumulative'] = test_df['count'].cumsum()
    matched_test_df['cumulative'] = matched_test_df['count'].cumsum()
    long_test_df['cumulative'] = long_test_df['count'].cumsum()
    left_out_df['cumulative'] = left_out_df['count'].cumsum()
    plt.tick_params(axis='both', bottom='on', left='on')
    plt.plot(train_df.submission_time, train_df.cumulative, label='train_df', lw=2)
    plt.plot(test_df.submission_time, test_df.cumulative, label='test_df', lw=2)
    plt.plot(matched_test_df.submission_time, matched_test_df.cumulative, label='matched_test_df', lw=2)
    plt.plot(long_test_df.submission_time, long_test_df.cumulative, label='long_test_df', lw=2)
    plt.plot(left_out_df.submission_time, left_out_df.cumulative, label='left_out_df', lw=2)
    #plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.ylabel('Cumulative count')
    plt.xticks(rotation=60)
    plt.savefig('figs/submissiondate.png', bbox_inches='tight')


def sub_time_by_cov(ax):
    d = DatasetStats()
    data = d.meta_data
    data = data.sort_values('submission_time', ascending=True)
    data['count'] = 1
    data_positive = data[data['test_result'] == 'Positive']
    data_negative = data[data['test_result'] == 'Negative']
    data_positive['cumulative'] = data_positive['count'].cumsum()
    data_negative['cumulative'] = data_negative['count'].cumsum()
    ax.tick_params(axis='both', bottom='on', left='on')
    ax.plot(data_positive.submission_time, data_positive.cumulative, label='COVID Positive', lw=2, color='darkred')
    ax.plot(data_negative.submission_time, data_negative.cumulative, label='COVID Negative', lw=2, color='mediumblue')
    #plt.yscale('log')
    #ax.grid()
    ax.legend()
    ax.set_ylabel('Cumulative count')
    #ax.set_xticks(ax.get_xticklabels(), rotation=60)
    ax.tick_params(axis='x', rotation=60)
if __name__ == '__main__':

    fig, ax = plt.subplots(111)
    sub_time_by_cov(ax)
    plt.savefig('figs/subtimee.pdf')
