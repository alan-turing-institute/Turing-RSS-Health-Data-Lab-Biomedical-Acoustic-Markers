import pandas as pd
import argparse
import numpy as np
import yaml
import swifter
import os, yaml, subprocess, glob, csv
from tqdm import tqdm
import pickle
from botocore import UNSIGNED
from botocore.config import Config
import io
import boto3
import sys
sys.path.append('../')


from sklearn.model_selection import KFold, train_test_split
from utils.savetos3 import save_to_s3

#for text to speech sensitive analysis
import torch
import torchaudio
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from sentence_transformers import SentenceTransformer
#from sklearn.covariance import EllipticEnvelope cannot use as rep too high dim
from sklearn.svm import OneClassSVM

class DatasetStats():
    '''
    class to create train test splits for paper and handle the meta data file and some sensitivity analysis
    for attributes please refer the argpars func at bottom of script
    '''
    RANDOM_SEED = 42

    def __init__(self, create_meta=False, text_extract=False, embed_sentence=False, outlier_detector=False, present_outliers=False, create_matched_validation=False):
        try:
            with open('config.yml', 'r')as conf:
                self.PATHS = yaml.safe_load(conf)
        except FileNotFoundError as err:
            raise ValueError(f'You need to specify your local paths to the data and meta data: {err}')
        self.bucket_meta = self.get_bucket(self.PATHS['meta_bucket'])
        self.bucket_audio = self.get_bucket(self.PATHS['audio_bucket'])
        self.create_meta = create_meta
        self.create_matched_validation = create_matched_validation
        if create_meta:
            print('creating the splits and adding to meta data')
            self.meta_data, self.train, self.test, self.long, self.orig_train, self.orig_test = self.load_train_test_splits()
            self.train, self.validation = self.create_validation(self.train)
            self.orig_train, self.orig_validation = self.create_validation(self.orig_train)
            self.create_naive_splits()
            self.merge_splits_meta()
        elif create_matched_validation:
            self.meta_data = pd.read_csv(self.get_file(
                                            'BAMstudy2022-prep/meta_data_with_splits_old_format.csv',
                                            self.bucket_meta))
            self.meta_data.sort_values('participant_identifier', inplace=True)
            print(self.meta_data.participant_identifier)
            self.create_match_val()
        else:
            print('Loading saved metafile')
            self.meta_data = pd.read_csv(self.get_file(
                                            'BAMstudy2022-prep/participant_metadata_160822.csv',
                                            self.bucket_meta))
            self.splits = pd.read_csv(self.get_file(
                                            'BAMstudy2022-prep/train_test_splits_160822.csv',
                                            self.bucket_meta))
            self.audio = pd.read_csv(self.get_file(
                                            'BAMstudy2022-prep/audio_metadata_160822.csv',
                                            self.bucket_meta))
            # Temporary measure while dataset is still on s3
            self.s3_lookup = pd.read_csv(self.get_file(
                                            'BAMstudy2022-prep/audio_lookup.csv',
                                            self.bucket_meta))
            self.s3_lookup.rename(columns={'exhalation_url_url': 'exhalation_url'}, inplace=True)
            self.sentence_lookup = self.s3_lookup[['sentence_file_name', 'sentence_url']]
            self.cough_lookup = self.s3_lookup[['cough_file_name', 'cough_url']]
            self.three_cough_lookup = self.s3_lookup[['three_cough_file_name', 'three_cough_url']]
            self.exhalation_lookup = self.s3_lookup[['exhalation_file_name', 'exhalation_url']]


            self.meta_data = pd.merge(self.meta_data, self.splits, on='participant_identifier')
            self.meta_data = pd.merge(self.meta_data, self.audio, on='participant_identifier')
            self.meta_data = self.meta_data.merge(
                    self.s3_lookup,
                    left_on=['exhalation_file_name', 'sentence_file_name', 'cough_file_name','three_cough_file_name'],
                    right_on=['exhalation_file_name', 'sentence_file_name', 'cough_file_name', 'three_cough_file_name'],
                    how='left')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stats()

    def get_bucket(self, bucket_name):
        s3_resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED), region_name='eu-west-2') 
        return s3_resource.Bucket(bucket_name)


    def get_file(self, path, bucket):
        return io.BytesIO(bucket.Object(path).get()['Body'].read())

    def load_train_test_splits(self):
        '''
        Loads the train and test barcode splits and the corresponding meta_data
        '''

        meta_data = pd.read_csv(self.get_file(
                                        self.PATHS['meta'],
                                        self.bucket_meta))

        train_test = pd.read_pickle(self.get_file(
                                        self.PATHS['splits_rebalanced'],
                                        self.bucket_meta))
        train_test_original = pd.read_pickle(self.get_file(
                                        self.PATHS['splits_original'],
                                        self.bucket_meta))

        return meta_data, train_test['train'], train_test['test'], [0] + train_test['longitudinal'], train_test_original['train'], train_test_original['test'] 

    def create_validation(self, train):
        '''
        create validation set
        inputs:
            train (list): the training set to split to create validation and training
        '''
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.RANDOM_SEED)
        folds = [[train[idx] for idx in v] for (t, v) in kfold.split(train)]
        fold = 1 # as the datasets is on the larger size no need for k fold cross validation
        new_train = [instance for instance in train if instance not in folds[fold-1]]
        validation = [instance for instance in train if instance in folds[fold-1]]
        assert not any(x in validation for x in new_train), 'there is cross over between train and validation'

        train = new_train

        return train, validation

    def create_naive_splits(self):
        '''
        creates the naive or random splits
        '''
        data = self.train + self.validation + self.test + self.long
        train, dev_test = train_test_split(
                    data,
                    test_size=0.3,
                    random_state=self.RANDOM_SEED)
        devel, test = train_test_split(
                    dev_test,
                    test_size=0.5,
                    random_state=self.RANDOM_SEED)

        self.naive_train, self.naive_validation, self.naive_test = train, devel, test

        assert not any(x in self.naive_validation for x in self.naive_train), 'there is cross over between naive train and validation'
        assert not any(x in self.naive_test for x in self.naive_train), 'there is cross over between naive train and test'
        assert not any(x in self.naive_validation for x in self.naive_test), 'there is cross over between naive test and validation'


    def merge_splits_meta(self):
        self.meta_data['splits'] = self.meta_data['participant_identifier'].apply(self.split_des)
        self.meta_data['original_splits'] = self.meta_data['participant_identifier'].apply(self.split_des_original)
        self.meta_data['naive_splits'] = self.meta_data['participant_identifier'].apply(self.split_des_naive)

    def split_des(self, x):

        if x in self.train:
            return 'train'
        elif x in self.validation:
            return 'val'
        elif x in self.test:
            return 'test'
        elif x in self.long:
            return 'long'
        else:
            return 'Not used'

    def split_des_original(self, x):
        if x in self.orig_train:
            return 'train'
        elif x in self.orig_validation:
            return 'val'
        elif x in self.orig_test:
            return 'test'
        elif x in self.long:
            return 'long'
        else:
            return 'Not used'
    
    def split_des_naive(self, x):
        if x in self.naive_train:
            return 'train'
        elif x in self.naive_validation:
            return 'val'
        elif x in self.naive_test:
            return 'test'
        else:
            return 'Not used'

    def split_des_matched(self, x):
        if x in self.matched_train:
            return 'matched_train'
        elif x in self.matched_validation:
            return 'matched_validation'
        else:
            return 'Not used'
    
    def split_des_matched_original(self, x):
        if x in self.original_matched_train:
            return 'matched_train'
        elif x in self.original_matched_validation:
            return 'matched_validation'
        else:
            return 'Not used'

    def stats(self):
        '''
        prints an array of stats for the dataset
        '''
        train = self.meta_data[self.meta_data['splits'] == 'train']
        self.train_pos = train[train['covid_test_result'] == 'Positive']
        self.train_neg = train[train['covid_test_result'] == 'Negative']

        val = self.meta_data[self.meta_data['splits'] == 'val']
        val_pos = val[val['covid_test_result'] == 'Positive']
        val_neg = val[val['covid_test_result'] == 'Negative']

        test = self.meta_data[self.meta_data['splits'] == 'test']
        self.test_pos = test[test['covid_test_result'] == 'Positive']
        self.test_neg = test[test['covid_test_result'] == 'Negative']

        long= self.meta_data[self.meta_data['splits'] == 'long']
        self.long_pos = long[long['covid_test_result'] == 'Positive']
        self.long_neg = long[long['covid_test_result'] == 'Negative']
        if self.create_matched_validation or not self.create_validation:
            match_train = self.meta_data[self.meta_data['matched_train_splits'] == 'matched_train']
            match_train_pos = match_train[match_train['covid_test_result'] == 'Positive']
            match_train_neg = match_train[match_train['covid_test_result'] == 'Negative']
            
            match_validation = self.meta_data[self.meta_data['matched_train_splits'] == 'matched_validation']
            match_validation_pos = match_validation[match_validation['covid_test_result'] == 'Positive']
            match_validation_neg = match_validation[match_validation['covid_test_result'] == 'Negative']
            
            test_matched = self.meta_data[self.meta_data['in_matched_rebalanced_test'] == True]
            self.test_matched_pos = test_matched[test_matched['covid_test_result'] == 'Positive']
            self.test_matched_neg = test_matched[test_matched['covid_test_result'] == 'Negative']

            long_test_matched = self.meta_data[self.meta_data['in_matched_rebalanced_long_test'] == True]
            self.long_test_matched_pos = long_test_matched[long_test_matched['covid_test_result'] == 'Positive']
            self.long_test_matched_neg = long_test_matched[long_test_matched['covid_test_result'] == 'Negative']
        
        print('Rebalanced Splits')
        print(f"total: {len(train) + len(val) + len(test) + len(long)}")
        print(f"train: {len(train)} positive: {len(self.train_pos)} negative {len(self.train_neg)}")
        print(f"val: {len(val)} positive: {len(val_pos)} negative {len(val_neg)}")
        print(f"train+val: {len(train) + len(val)} positive: {len(self.train_pos) + len(val_pos)} negative {len(self.train_neg)+ len(val_neg)}")
        print(f"test: {len(test)} positive: {len(self.test_pos)} negative {len(self.test_neg)}")
        print(f"long: {len(long)} positive: {len(self.long_pos)} negative {len(self.long_neg)}")
        if self.create_matched_validation or not self.create_validation:
            print(f"matched test: {len(test_matched)} positive: {len(self.test_matched_pos)} negative {len(self.test_matched_neg)}")
            print(f"matched train: {len(match_train)} positive: {len(match_train_pos)} negative {len(match_train_neg)}")
            print(f"matched val: {len(match_validation)} positive: {len(match_validation_pos)} negative {len(match_validation_neg)}")
            print(f"matched train+val: {len(match_train) + len(match_validation)} positive: {len(match_train_pos) + len(match_validation_pos)} negative {len(match_train_neg)+ len(match_validation_neg)}")
            print(f"matched long: {len(long_test_matched)} positive: {len(self.long_test_matched_pos)} negative {len(self.long_test_matched_neg)}")

        train = self.meta_data[self.meta_data['original_splits'] == 'train']
        train_pos = train[train['covid_test_result'] == 'Positive']
        train_neg = train[train['covid_test_result'] == 'Negative']

        val = self.meta_data[self.meta_data['original_splits'] == 'val']
        val_pos = val[val['covid_test_result'] == 'Positive']
        val_neg = val[val['covid_test_result'] == 'Negative']

        test = self.meta_data[self.meta_data['original_splits'] == 'test']
        test_pos = test[test['covid_test_result'] == 'Positive']
        test_neg = test[test['covid_test_result'] == 'Negative']

        long= self.meta_data[self.meta_data['original_splits'] == 'long']
        long_pos = long[long['covid_test_result'] == 'Positive']
        long_neg = long[long['covid_test_result'] == 'Negative']
        
        if self.create_matched_validation or not self.create_validation:
            test_matched = self.meta_data[self.meta_data['in_matched_original_test'] == True]
            test_matched_pos = test_matched[test_matched['covid_test_result'] == 'Positive']
            test_matched_neg = test_matched[test_matched['covid_test_result'] == 'Negative']
            
            match_train = self.meta_data[self.meta_data['matched_original_train_splits'] == 'matched_train']
            match_train_pos = match_train[match_train['covid_test_result'] == 'Positive']
            match_train_neg = match_train[match_train['covid_test_result'] == 'Negative']
            
            match_validation = self.meta_data[self.meta_data['matched_original_train_splits'] == 'matched_validation']
            match_validation_pos = match_validation[match_validation['covid_test_result'] == 'Positive']
            match_validation_neg = match_validation[match_validation['covid_test_result'] == 'Negative']
        print('Original splits')
        print(f"total: {len(train) + len(val) + len(test) + len(long)}")
        print(f"train: {len(train)} positive: {len(train_pos)} negative {len(train_neg)}")
        print(f"val: {len(val)} positive: {len(val_pos)} negative {len(val_neg)}")
        print(f"train+val: {len(train) + len(val)} positive: {len(train_pos) + len(val_pos)} negative {len(train_neg)+ len(val_neg)}")
        print(f"test: {len(test)} positive: {len(test_pos)} negative {len(test_neg)}")
        print(f"long: {len(long)} positive: {len(long_pos)} negative {len(long_neg)}")
        if self.create_matched_validation or not self.create_validation:
            print(f"matched test: {len(test_matched)} positive: {len(test_matched_pos)} negative {len(test_matched_neg)}")
            print(f"matched train: {len(match_train)} positive: {len(match_train_pos)} negative {len(match_train_neg)}")
            print(f"matched val: {len(match_validation)} positive: {len(match_validation_pos)} negative {len(match_validation_neg)}")
            print(f"matched train+val: {len(match_train) + len(match_validation)} positive: {len(match_train_pos) + len(match_validation_pos)} negative {len(match_train_neg)+ len(match_validation_neg)}")

        train = self.meta_data[self.meta_data['naive_splits'] == 'train']
        train_pos = train[train['covid_test_result'] == 'Positive']
        train_neg = train[train['covid_test_result'] == 'Negative']

        val = self.meta_data[self.meta_data['naive_splits'] == 'val']
        val_pos = val[val['covid_test_result'] == 'Positive']
        val_neg = val[val['covid_test_result'] == 'Negative']

        test = self.meta_data[self.meta_data['naive_splits'] == 'test']
        self.naive_test_pos = test[test['covid_test_result'] == 'Positive']
        self.naive_test_neg = test[test['covid_test_result'] == 'Negative']

        print('Naive stats:') 
        print('Totals on merging naive_splits with meta')
        print(f"total: {len(train) + len(val) + len(test)}")
        print(f"train: {len(train)} positive: {len(train_pos)} negative {len(train_neg)}")
        print(f"train+val: {len(train) + len(val)} positive: {len(train_pos) + len(val_pos)} negative {len(train_neg)+ len(val_neg)}")
        print(f"val: {len(val)} positive: {len(val_pos)} negative {len(val_neg)}")
        print(f"test: {len(test)} positive: {len(self.naive_test_pos)} negative {len(self.naive_test_neg)}")

    def text_extractor(self, modality):
        '''
        speech to text extraction
        inputs:
            modality (str): respiratory audio modality
        '''
        #set up model and processor
        print('setting up model and processor')
        model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
        processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
        print(f'now iterating through meta_data for {modality}')
        # note I am not batching and sending to gpu so ppl with only cpu can run efficiently
        self.meta_data[f'Participants said: ({modality})'] = self.meta_data[modality].swifter.apply(lambda x: self.text_to_speech(model, processor, x, modality))

    def text_to_speech(self, model, processor, path_to_audio, modality):
        '''
        speech to text for one file. Loads and transcribes
        inputs:
            model (ML transcription model): preprocess audio
            processor (ML transciption model): ML computation 
            path_to_audio (str): path to audio file
            modality (str): respiratory audio modality
        '''
        try:
            filename = self.get_file(path_to_audio, self.bucket_audio)
        except:
            print('not possible to load')
            return 'Not possible to load audio'
        try:
            signal, sr = torchaudio.load(filename)
        except:
            print(f'not possible to load {filename}')
            return 'Not possible to load audio'
        if (signal.size()[0] < 1 or signal.size()[1] < 1):
            return 'Not possible to load audio'
        transform = torchaudio.transforms.Resample(sr, 16000)
        signal = transform(signal)
        sr = 16000
        inputs = processor(signal[0].numpy(), sampling_rate=sr, return_tensors="pt")
        generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

        transcription = processor.batch_decode(generated_ids)
        print(f'''Processing file: {path_to_audio.rsplit('/',1)[-1]} Transcription: {transcription}''')
        return transcription

    def embed_sentences(self, modality):
        '''
        create vector representation of transcribed sentence
        '''
        print('Using', self.device)
        model = SentenceTransformer('all-mpnet-base-v2',
                device=self.device)
        sentences = self.meta_data[f'Participants said: ({modality})'].tolist()
        sentences = [sentence[0] for sentence in sentences]
        print('Embedding sentences!')
        embeddings = model.encode(sentences,
                device=self.device,
                show_progress_bar=True,
                batch_size=32)
        self.meta_data[f'Sentence Embedding ({modality})'] = pd.Series([i for i in embeddings])

    def outlier_detector(self, modality):
        '''
        Fit an SVM to the distribution of vector setence representations and Identify the outliers. This outliers are then manually checked
        '''
        print('Predicting the outliers based on sentence embeddings')
        X = np.concatenate([np.expand_dims(i, axis=0) for i in self.meta_data[f'Sentence Embedding ({modality})'].tolist()], axis=0)
        clf = OneClassSVM(gamma='auto') 
        self.meta_data[f'Sentence Outlier ({modality})'] = pd.Series(clf.fit_predict(X))
        self.meta_data[f'Sentence Outlier Scores ({modality})'] = pd.Series(clf.score_samples(X))
        outliers = self.meta_data[self.meta_data[f'Sentence Outlier ({modality})'] == -1]
        print(outliers[f'Participants said: ({modality})'])

    def present_outliers(self, modality):
        sorted_meta = self.meta_data.sort_values(f'Sentence Outlier Scores ({modality})')
        sorted_meta[f'Participants said: ({modality})'] = sorted_meta[f'Participants said: ({modality})'].apply(lambda x: x[0])
        print(sorted_meta[[f'Participants said: ({modality})', f'Sentence Outlier Scores ({modality})']])
        sorted_meta = sorted_meta.drop_duplicates(subset=f'Participants said: ({modality})')
        myfile = open(f'{modality}_outliers.txt', 'w')
        for i, (index, row) in enumerate(sorted_meta.iterrows()):
            if i == 1000:
                break
            myfile.write(f'''
File: {row[{modality}]}
Outlier Score: {row[f'Sentence Outlier Scores ({modality})']}
Sentence: {row[f'Participants said: ({modality})']}''')
        myfile.close()

    def create_match_val(self):
        '''
        creating validation split - disjoint across strata
        '''
        self.matched_train, self.matched_validation = self.create_validation(
                self.meta_data[self.meta_data['stratum_matched_rebalanced_train'].notnull()].stratum_matched_rebalanced_train.unique().tolist()
                )
        
        self.original_matched_train, self.original_matched_validation = self.create_validation(
                self.meta_data[self.meta_data['stratum_matched_original_train'].notnull()].stratum_matched_original_train.unique().tolist()
                )

        self.meta_data['matched_train_splits'] = self.meta_data['stratum_matched_rebalanced_train'].apply(self.split_des_matched)
        self.meta_data['matched_original_train_splits'] = self.meta_data['stratum_matched_original_train'].apply(self.split_des_matched)


    def save_file(self):
        '''
        saves the constructed meta file to s3. This should only be called once - after this the
        meta data table should be loaded from s3
        '''
        if self.create_meta:

            save_to_s3(self.meta_data[['participant_identifier', 'splits', 'original_splits', 'naive_splits']],
                    filename='train_test_splits_stage2.csv', 
                    location='BAMstudy2022-prep/',
                    paths=self.PATHS)

        elif self.create_matched_validation:

            save_to_s3(self.meta_data[
                ['participant_identifier',
                    'splits',
                    'original_splits',
                    'naive_splits',
                    'in_matched_original_test',
                    'stratum_matched_original_test',
                    'in_matched_rebalanced_test',
                    'stratum_matched_rebalanced_test',
                    'in_matched_original_long_test',
                    'stratum_matched_original_long_test',
                    'in_matched_rebalanced_long_test',
                    'stratum_matched_rebalanced_long_test',
                    'in_matched_original_train',
                    'stratum_matched_original_train',
                    'in_matched_rebalanced_train',
                    'stratum_matched_rebalanced_train',
                    'matched_train_splits',
                    'matched_original_train_splits']], 
                filename='train_test_splits_stage4.csv', 
                location='BAMstudy2022-prep/',
                paths=self.PATHS)
        else:
            print('We do not provide reproducible steps for the sensitivity analysis as sensitive content has already been removed therefore no save')



def load_args():
    parser = argparse.ArgumentParser(
        description="Class to create/load the ciab dataset")
    parser.add_argument("--create_meta", type=str, help="Do you want to create the train test splits from scratch? or load the provided splits?", choices=['yes', 'no'], default='no')
    parser.add_argument("--create_matched_validation", type=str, help="Do you want to re create the matched validation set?", choices=['yes', 'no'], default='no')
    parser.add_argument("--sentence_similarity", type=str, help="Do you want to run sentence similarity caluculation?", choices=['yes', 'no'], default='no')
    args = parser.parse_args()
    args.create_meta = True if args.create_meta == 'yes' else False
    args.create_matched_validation = True if args.create_matched_validation == 'yes' else False
    if args.sentence_similarity == 'yes':
        args.sentence_similarity == True
        args.text_extract = True
        args.embed_sentence = True
        args.outlier_detector = True
    else:
        args.text_extract = False
        args.embed_sentence = False
        args.outlier_detector = False
        args.sentence_similarity == True


    return args

if __name__ == '__main__':
    args = load_args()
    d = DatasetStats(
            create_meta=args.create_meta, 
            text_extract=args.text_extract, 
            embed_sentence=args.embed_sentence, 
            outlier_detector=args.outlier_detector, 
            create_matched_validation=args.create_matched_validation)
    if args.sentence_similarity:
        POSSIBLE_MODALITIES = ['sentence_url',
                           'exhalation_url',
                           'cough_url',
                           'three_cough_url']

        for modality in POSSIBLE_MODALITIES:
            d.text_extract(modality)
            d.embed_sentences(modality)
            d.outlier_detector(modality)
            d.present_outliers(modality)

    d.save_file()
