import csv
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import librosa, librosa.display
import pprint
import io
import matplotlib.pyplot as plt
import IPython.display as ipd
import pandas as pd
import os
import sys
from vggish.vggish_input import waveform_to_examples
import numpy as np
from tqdm import tqdm
import config

import warnings
warnings.filterwarnings('ignore')



def get_file(path, bucket):
    ''' Retrieve file from s3 bucket.
        Input: `path` Path to file from bucket
               `bucket` Bucket location
        Output: file object
    '''
    return io.BytesIO(bucket.Object(path).get()['Body'].read())



def get_file_label_list_for_split(split_type, data_type, modality, df_full):
    ''' Retrieve a list of files and corresponding labels (Positive: COVID positive, Negative: COVID negative)
        for each experiment as defined in the paper. The output list is given as a function of
        pre-defined input and split types which match from the study metadata dataframe.
        
        
        Input: `split_type`: `naive_splits`, `standard`, `in_matched_rebalanced`
               `data_type`: `train`, `val`, `test`.
               `modality`: Column name for saving features. With the default dataframe `df_full`:
                        ['sentence_url', 'exhalation_url_url', 'cough_url', 'three_cough_url'].
                        These refer to spoken sentece, forced exhalation ('ha' sound), a single
                        cough or three coughs.
               `df_full`: Pandas dataframe containing all metadata. See main execution for details.
        Output: `file_list`: list of file paths to be loaded downstream
                `label_list`: list of same length of `file_list` containing COVID positive/negative label
    '''
    
    if split_type == 'naive_splits':
        if data_type == 'long':
            return None, None
        else:
            file_list = df_full[df_full[split_type] == data_type][modality].to_list()
            label_list = df_full[df_full[split_type] == data_type]["covid_test_result"].to_list()
    elif split_type == 'standard':
        file_list = df_full[df_full['splits'] == data_type][modality].to_list()
        label_list = df_full[df_full['splits'] == data_type]["covid_test_result"].to_list()
    elif split_type == 'matched':
        col_name = 'in_matched_rebalanced'
        
        if data_type == 'train' or data_type == 'val':
            df_data = df_full[df_full[col_name + '_' + 'train']]
            file_list = df_data[df_data['splits'] == data_type][modality].to_list()
            label_list = df_data[df_data['splits'] == data_type]["covid_test_result"].to_list()
        elif data_type == 'long':
            df_data = df_full[df_full[col_name + '_' + 'long_test']]
            file_list = df_data[modality].to_list()
            label_list = df_data["covid_test_result"].to_list()
        else:
            file_list = df_full[df_full[col_name + '_' + data_type]][modality].to_list()
            label_list = df_full[df_full[col_name + '_' + data_type]]["covid_test_result"].to_list()  
    else:
        raise ValueError('Split type not in defined options. Please select from `naive_splits`,`standard` or `in_matched_rebalanced`.')
    if not (split_type == 'naive_splits' and data_type == 'long'):
        print('Split type', split_type, 'data_type', data_type, 'modality',
              modality, 'N_x =', len(file_list), 'N_y = ', len(label_list))
    
    return file_list, label_list



def get_feat(audio_url_list, label_list, bucket_name, modality, feat_dir = config.feat_dir,
             n_samples_per_pickle=3000, feat_name=None):
    '''Create features for list of audio urls. Files are dumped in pickles to disk in parts due
    to the size of feature arrays exceeding limits. Function returns only bugged keys if they
    cannot be processed.
    
    Inputs: `audio_url_list`: full list of audio URLs to process
            `label_list`: corresponding label_list with values to create labels
            `bucket_name`: Name of s3 bucket
            `modality`: Column name for saving features. By default in:
                        ['sentence_url', 'exhalation_url_url', 'cough_url', 'three_cough_url']
            `feat_dir`: Location to save as defined in config
            `n_samples_per_pickle`: Number of data points to save per pickle. 3000 default will create
                                  pickles of approximately 1GB each.
            `feat_name`: Name of pickle, important for loading data downstream. Highly recommended to
                       leave as default.
                       
     Outputs: Features are saved as specified to `feat_dir`. Function has no return value.
     '''
    
    s3_resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED), region_name='eu-west-2')
    bucket = s3_resource.Bucket(bucket_name)    
    
    n = len(audio_url_list)//n_samples_per_pickle
    
    for i in range(n+1):  # Begin to extract features and save them in n_samples_per_pickle
        audio_url_subset = audio_url_list[i*n_samples_per_pickle:(i+1)*n_samples_per_pickle]
        label_list_subset = label_list[i*n_samples_per_pickle:(i+1)*n_samples_per_pickle]
        feat_list = []
        y_list = []
        
        for index, audio_url in tqdm(enumerate(audio_url_subset)):
            signal, rate = librosa.load(get_file(audio_url, bucket), sr=None) 
            # Load with librosa without re-sampling
            
            # Code own features here if desired
            # Edit any options of vggish features in vggish/vggish_params.py
            features = waveform_to_examples(signal, sample_rate=rate, return_tensor=False)
            feat_list.append(features)
            label = label_list_subset[index]
            
            if label == 'Positive':
                y_list.append([1] * len(features))
            elif label == 'Negative':
                y_list.append([0] * len(features))
            else:
                raise ValueError('Label type not recognised, must be in `Positive`, `Negative`')

        np.save(os.path.join(feat_dir, feat_name + '_feat_list_' + modality + '_' + str(i) + '.npy'), feat_list)
        np.save(os.path.join(feat_dir, feat_name + '_label_list_' + modality + '_' + str(i) + '.npy'), y_list)
        print('Saving features and labels for partition:', i, 'of', n)
    return




if __name__ == '__main__':
    
    # Read relevant metadata dataframes for merging and extracting audio and labels
    s3_resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED), region_name='eu-west-2')
    bucket = s3_resource.Bucket(config.bucket_name)
    
    df_meta = pd.read_csv(get_file(config.audio_metadata, bucket))
    df_train_test_splits = pd.read_csv(get_file(config.train_test_split, bucket))
    df_audio_url = pd.read_csv(get_file(config.audio_lookup, bucket))
    df_participant = pd.read_csv(get_file(config.participant_metadata, bucket))

    # Merge on participant_identifier, then do to_list() of the filenames/urls from audio lookup

    df_full = df_train_test_splits.merge(df_meta, on='participant_identifier').merge(df_participant,
                                         on ='participant_identifier').merge(df_audio_url, on='cough_file_name')

    for split_type in ['naive_splits', 'matched', 'standard']:
        for data_type in ['train', 'val', 'test', 'long']:
            for modality in ['sentence_url', 'exhalation_url_url', 'cough_url', 'three_cough_url']:
                file_list, label_list = get_file_label_list_for_split(split_type, data_type, modality, df_full)
                if file_list:
                    print('Extracting features now')
                    get_feat(file_list, label_list, config.bucket_name, modality,
                     feat_name= split_type + '_' + data_type)


