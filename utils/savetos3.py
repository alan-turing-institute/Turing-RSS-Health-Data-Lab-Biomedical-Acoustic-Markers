'''
File to save predictions to s3
author: Harry Coppock
Qs: harry.coppock@imperial.ac.uk
'''
import pandas as pd
import re
import sys
import yaml
sys.path.append('../')
from ssast_ciab.src.finetune.ciab.prep_ciab import PrepCIAB
from sklearn import metrics
import numpy as np
import boto3

def load_predictions(path):
    df = pd.read_csv(path, names=['Negative', 'Positive', 'id'])
    return df

def save_to_s3(
        df, 
        filename='ss_predicts_matched_train_matched_test.csv',
        location='audio_sentences_for_matching/',
        paths=None):
    '''
    saves dataframe as csv to s3
    '''
    df.to_csv(filename)
    s3 = boto3.client('s3')

    with open(filename, 'rb') as f:
        s3.put_object(Bucket=paths['meta_bucket'],
                      Key=f'{location}{filename}',
                      Body=f
                     )

def format_id(id):
    id = re.search('[^/]+$', id)
    return id.group(0)

def check_scores_sane(df):
    '''
    George this func maybe useful to you for repeating the 3 classification metrics on each strata.
    args: pd.DataFrame - containing predictions in logit form for both Negative and Positive cases
    '''
    df['test_result'] = df['test_result'].apply(lambda x: 0 if x == 'Negative' else 1)
    uar = metrics.recall_score(
            df['test_result'].to_numpy(),
            np.argmax(df[['Negative', 'Positive']].to_numpy(), 1),
            average='macro')

    avg_precision = metrics.average_precision_score(
            df['test_result'].to_numpy(),
            df['Positive'].to_numpy(),
            average=None)

    auc = metrics.roc_auc_score(
            df['test_result'].to_numpy(),
            df['Positive'].to_numpy(),
            average=None)
    print(df)
    print(uar)
    print(avg_precision)
    print(auc)

def load_meta():
    extract = PrepCIAB()
    return extract.meta

def merge_features_meta(features, meta_data):
    print(features.columns)
    print(meta_data.columns)
    return features.merge(meta_data,
                            how='inner',
                             on='audio_sentence')
def main(preds_path, save_name):

    try:
        with open('config.yml', 'r')as conf:
            paths = yaml.safe_load(conf)
    except FileNotFoundError as err:
        raise ValueError(f'You need to specify your local paths to the data and meta data: {err}')
    preds = load_predictions(preds_path)
    preds.rename(columns={'id':'audio_sentence'}, inplace=True)
    meta = load_meta()
    df = merge_features_meta(preds, meta)
    check_scores_sane(df) #checking scores are the same 
    save_to_s3(df, save_name, paths)

if __name__ == '__main__':
    preds_path = '/workspace/ssast_ciab/src/finetune/ciab/exp/inference/train_long_test/predictionstrain_long_test/predictions_0.csv'
    main(preds_path, 'ss_predicts_standard_train_long_test.csv')
    preds_path = '/workspace/ssast_ciab/src/finetune/ciab/exp/inference/train_test/predictionstrain_test/predictions_0.csv'
    main(preds_path, 'ss_predicts_standard_train_test.csv')
    preds_path = '/workspace/ssast_ciab/src/finetune/ciab/exp/inference/train_validation/predictionstrain_validation/predictions_0.csv'
    main(preds_path, 'ss_predicts_standard_train_val.csv')
    preds_path = '/workspace/ssast_ciab/src/finetune/ciab/exp/inference/train_train/predictionstrain_train/predictions_0.csv'
    main(preds_path, 'ss_predicts_standard_train_train.csv')
    preds_path = '/workspace/ssast_ciab/src/finetune/ciab/exp/inference/train_matched_test/predictionstrain_matched_test/predictions_0.csv'
    main(preds_path, 'ss_predicts_standard_train_matched_test.csv')
    preds_path = '/workspace/ssast_ciab/src/finetune/ciab/exp/inference/train_long_matched_test/predictionstrain_long_matched_test/predictions_0.csv'
    main(preds_path, 'ss_predicts_standard_train_long_matched_test.csv')
    preds_path = '/workspace/ssast_ciab/src/finetune/ciab/exp/inference/matched_train_matched_test/predictionsmatched_train_matched_test/predictions_0.csv'
    main(preds_path, 'ss_predicts_matched_train_matched_test.csv')
    preds_path = '/workspace/ssast_ciab/src/finetune/ciab/exp/inference/matched_train_long_matched_test/predictionsmatched_train_long_matched_test/predictions_0.csv'
    main(preds_path, 'ss_predicts_matched_train_long_matched_test.csv')
