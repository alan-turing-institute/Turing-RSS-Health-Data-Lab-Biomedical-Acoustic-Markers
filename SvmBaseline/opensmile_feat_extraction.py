'''
Code for extracting opensmile features for the CIAB dataset
Author: Harry Coppock
Qs to: harry.coppock@imperial.ac.uk
Tests: Once run please run the unit tests in /home/ec2-user/SageMaker/jbc-cough-in-a-box/test
'''
import pandas as pd
import os, yaml, subprocess, glob, csv
from tqdm import tqdm
import pickle
from botocore import UNSIGNED
from botocore.config import Config
import io
import boto3
import librosa
import soundfile as sf
import sys

sys.path.append('../')

from utils.dataset_stats import DatasetStats


class ExtractOpensmile(DatasetStats):

    '''
    Handles the extraction of openSMILE feature extraction
    Attributes:
        modality (str): which respiratory modality to extract for
        symp_clf (bool): do you want to extract features for symptoms or covid classification

    '''
    POSSIBLE_MODALITIES = ['exhalation_url', 
                    'cough_url',
                    'three_cough_url',
                    'sentence_url']


    def __init__(self, modality='audio_three_cough_url', symp_clf=False):
        self.modality = self.check_modality(modality)
        super().__init__()
        # base directory for audio files
        self.symp_clf = symp_clf
        if self.symp_clf:
            print('extracting features for binary classification')
            self.output_base= f'./features/opensmile_symptoms_clf/{self.modality}'
        else:
            self.output_base= f'./features/opensmile_final/{self.modality}'
        self.feature_set = 'compare16/ComParE_2016.conf' #for now we will just look at compare feature sets
        self.load_splits()

    def load_splits(self):
        '''
        Loads the train and test barcode splits and the corresponding meta_data
        '''
        if not self.symp_clf:

            self.train = self.meta_data[self.meta_data['splits'] == 'train'].participant_identifier.tolist()
            self.val = self.meta_data[self.meta_data['splits'] == 'val'].participant_identifier.tolist()
            self.test = self.meta_data[self.meta_data['splits'] == 'test'].participant_identifier.tolist()
            self.long = self.meta_data[self.meta_data['splits'] == 'long'].participant_identifier.tolist()
            self.long_matched = self.meta_data[self.meta_data['in_matched_rebalanced_long_test'] == True].participant_identifier.tolist()

            self.matched_train = self.meta_data[self.meta_data['matched_train_splits'] == 'matched_train'].participant_identifier.tolist()
            self.matched_validation = self.meta_data[self.meta_data['matched_train_splits'] == 'matched_validation'].participant_identifier.tolist()
            self.matched_test = self.meta_data[self.meta_data['in_matched_rebalanced_test'] == True].participant_identifier.tolist()

            self.naive_train = self.meta_data[self.meta_data['naive_splits'] == 'train'].participant_identifier.tolist()
            self.naive_validation = self.meta_data[self.meta_data['naive_splits'] == 'val'].participant_identifier.tolist()
            self.naive_test = self.meta_data[self.meta_data['naive_splits'] == 'test'].participant_identifier.tolist()

            self.train_original = self.meta_data[self.meta_data['original_splits'] == 'train'].participant_identifier.tolist()
            self.val_original = self.meta_data[self.meta_data['original_splits'] == 'val'].participant_identifier.tolist()
            self.test_original = self.meta_data[self.meta_data['original_splits'] == 'test'].participant_identifier.tolist()

            self.matched_train_original = self.meta_data[self.meta_data['matched_original_train_splits'] == 'matched_train'].participant_identifier.tolist()
            self.matched_validation_original = self.meta_data[self.meta_data['matched_original_train_splits'] == 'matched_validation'].participant_identifier.tolist()
            self.matched_test_original = self.meta_data[self.meta_data['in_matched_original_test'] == True].participant_identifier.tolist()
        else:
            # we are only need to add new training sets, the evaluation sets will be the same!
            # we only train our symptoms classifier on covid negative individuals as otherwise signal could be from covid pos
            self.train = self.meta_data[(self.meta_data['splits'] == 'train') & (self.meta_data['covid_test_result'] == 'Negative')].participant_identifier.tolist()
            self.val = self.meta_data[(self.meta_data['splits'] == 'val') & (self.meta_data['covid_test_result'] == 'Negative')].participant_identifier.tolist()
            self.naive_train = self.meta_data[(self.meta_data['naive_splits'] == 'train') & (self.meta_data['covid_test_result'] == 'Negative')].participant_identifier.tolist()
            self.naive_validation = self.meta_data[(self.meta_data['naive_splits'] == 'val') & (self.meta_data['covid_test_result'] == 'Negative')].participant_identifier.tolist()


    def main(self):
        '''
        Extract features for each of the splits
        '''
        if not os.path.exists(self.output_base):
            os.makedirs(self.output_base)

        if not self.symp_clf:
            print('Beginining ciab train prepocessing')
            self.iterate_through_files(self.train, 'train')
            print('Beginining ciab validation prepocessing')
            self.iterate_through_files(self.val, 'val')
            print('Beginining ciab test prepocessing')
            self.iterate_through_files(self.test, 'test')
            print('Beginining ciab long test prepocessing')
            self.iterate_through_files(self.long, 'long_test')
            print('Beginining ciab long matched test prepocessing')
            self.iterate_through_files(self.long_matched, 'long_matched_test')
            print('Beginining ciab matched test prepocessing')
            self.iterate_through_files(self.matched_test, 'matched_test')
            print('Beginining ciab matched_train prepocessing')
            self.iterate_through_files(self.matched_train, 'matched_train')
            print('Beginining ciab matched_validation prepocessing')
            self.iterate_through_files(self.matched_validation, 'matched_validation')
            print('Beginining ciab naive train prepocessing')
            self.iterate_through_files(self.naive_train, 'naive_train')
            print('Beginining ciab naive validation prepocessing')
            self.iterate_through_files(self.naive_validation, 'naive_validation')
            print('Beginining ciab naive test prepocessing')
            self.iterate_through_files(self.naive_test, 'naive_test')
            print('Beginining ciab original train prepocessing')
            self.iterate_through_files(self.train_original, 'train_original')
            print('Beginining ciab original test prepocessing')
            self.iterate_through_files(self.val_original, 'val_original')
            print('Beginining ciab original test prepocessing')
            self.iterate_through_files(self.test_original, 'test_original')
            print('Beginining ciab matched test prepocessing')
            self.iterate_through_files(self.matched_test_original, 'matched_test_original')
            print('Beginining ciab matched_train prepocessing')
            self.iterate_through_files(self.matched_train_original, 'matched_train_original')
            print('Beginining ciab matched_validation prepocessing')
            self.iterate_through_files(self.matched_validation_original, 'matched_validation_original')

        else:

            print('Beginining ciab train prepocessing')
            self.iterate_through_files(self.train, 'train')
            print('Beginining ciab validation prepocessing')
            self.iterate_through_files(self.val, 'val')
            print('Beginining ciab naive train prepocessing')
            self.iterate_through_files(self.naive_train, 'naive_train')
            print('Beginining ciab naive validation prepocessing')
            self.iterate_through_files(self.naive_validation, 'naive_validation')
        

    def check_modality(self, modality):
        if modality not in self.POSSIBLE_MODALITIES:
            raise Exception(f"{modaliity} is not one of the recorded functionalities,\
                                 please choose from {self.POSSIBLE_MODALITIES}") 
        else:
            return modality   


    def import_opensmile(self):
        '''
        downloading from source now so this func is redundant for now
        '''
        try:
            import opensmile
        except ImportError:
            raise ImportError("Looks like you need to install opensmile,\
                                 run: pip install opensmile")
    def load_opensmile(self):
        '''
        downliading from source now so this func is redundant for now
        simply creates an instance of open smile, will add futher feature types
        if necessary
        '''
        return opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
            )

    

    def iterate_through_files(self, dataset, split='train'):
        '''
        For a split extract features for each recording
        dataset (list): list of participant identifiers in split
        split (str): which split to extract for - for naming 
        '''
        error_list = []
        outfile_base = os.path.join(self.output_base, split)
        outfile = f'{outfile_base}.csv'
        outfile_lld = f'{outfile_base}_lld.csv'
        for i, participant_identifier in enumerate(tqdm(dataset)):
            df_match = self.meta_data[self.meta_data['participant_identifier'] == participant_identifier]
            assert len(df_match) != 0, 'This unique code does not exist in the meta data file currently loaded - investigate!'
            try:
                filename = self.get_file(df_match[self.modality].iloc[0], self.bucket_audio)
            except:
                print(f"{df_match[self.modality].iloc[0]} not possible to load. From {df_match['submission_date']} Total so far: {len(error_list)}")
                error_list.append(df_match[self.modality].iloc[0])
                continue
            if not self.symp_clf:
                label = df_match['covid_test_result'].iloc[0]
            else:
                label = 'asymptomatic' if df_match['symptom_none'].iloc[0] == 1 else 'symptomatic'
            # Annoyingly I cannot configure opensmile to accept io based paths so I load and save
            # a temporary .wav file which is then loaded again by opensmile
            try:
                test_file, rate = librosa.load(filename, sr=None)
            except RuntimeError:
                print(f"{filename} not possible to load. From {df_match['submission_date']} Total so far: {len(error_list)}")
                error_list.append(filename) 
                continue
            sf.write(f"temp_files/librosa_{participant_identifier}.wav", test_file, rate)
            self.extract_opensmile(f"temp_files/librosa_{participant_identifier}.wav", outfile, label, participant_identifier)
            if i % 1000 == 0:
                self.delete_temp()
        self.delete_temp()
        with open(f'{outfile_base}.txt', "w") as output:
            output.write(str(error_list))

    def delete_temp(self,):
        '''
        Due to the mem constraints of the sagemaker notebooks reg delete the temp files
        '''
        cmd = 'rm -rf temp_files/*'
        os.system(cmd)


    def extract_opensmile(self, filename, outfile, label, participant_identifier):

        '''
        calls openSMILE package to extract the features of one file
        filename (str): name of file
        outfile (str): csv to save features to
        label (str): covid or symptom label
        pariticpant_identifier (str): unique participant identifier
        '''
        cmd = f'/workspace/SvmBaseline/opensmile/build/progsrc/smilextract/SMILExtract -noconsoleoutput -C ./opensmile/config/{self.feature_set} -I {filename} -N {participant_identifier} -class {label} -O {outfile}'# -lldarffoutput {outfile_lld} -timestamparfflld 0'
        os.system(cmd)

if __name__ == "__main__":
    
    modalities =  ['exhalation_url',
                   'cough_url',
                   'three_cough_url',
                    'sentence_url']
    
    for modality in modalities:
        print(f'*'*20)
        print(f'Beginging opensmile feature extraction for {modality}')
        print(f'*'*20)
    
        e = ExtractOpensmile(modality, symp_clf=False)
        print(e.meta_data)
        print(e.meta_data.columns) 
        e.main()
    
