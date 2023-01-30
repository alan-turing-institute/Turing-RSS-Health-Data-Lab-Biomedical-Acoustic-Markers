import boto3
import pandas
import io
from botocore import UNSIGNED
from botocore.config import Config
import pandas as pd
import numpy as np
import random
from collections import OrderedDict
import json
import requests
import os
from matplotlib import pyplot as plt
import seaborn as sns
import random
from configparser import ConfigParser

def get_file(path, bucket_name, region):
    s3_resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED), region_name=region)
    bucket = s3_resource.Bucket(bucket_name)
    return io.BytesIO(bucket.Object(path).get()['Body'].read())



class CIAB_Data():
    '''
    A class for loading and accessing data from the Cough In A Box study. No input parameters required
    '''
    def __init__(self):
        # Retrieve data config
        config_filepath = 's3_config.ini'
        config = ConfigParser()
        config.read(config_filepath)
        bucket_name = config['S3']['bucket']
        region = config['S3']['region']
        audio_path = config['S3']['audio_path']
        participant_path = config['S3']['participant_path']
        # Merge audio and participant data
        audio_data = pd.read_pickle(get_file(audio_path, bucket_name, region))
        participant_data = pd.read_pickle(get_file(participant_path, bucket_name, region))
        self.raw_study_data = audio_data.merge(participant_data).sort_values('participant_identifier')
        # Filter out submissions where a test result was not obtained with a PCR test
        filtered_data = self.raw_study_data.loc[self.raw_study_data['covid_test_method'].apply(lambda x:1 if x in ['LFT', 'LAMP', 'Unknown'] else 0) == 0]
        filtered_data['covid_test_result'] = filtered_data['covid_test_result'].apply(lambda x: np.NaN if x == 'Unknown/Void' else x)
        # Filter out submissions where the audio submission was completed before the test, or where the delay between test result and audio submission is >10 days
        filtered_data = filtered_data.loc[(filtered_data['submission_delay'] < 10) & (filtered_data['submission_delay'] >= 0)]
        # Filter out submissions where the PCR test was done in labs without reported testing issues
        filtered_data = filtered_data.loc[filtered_data['covid_test_lab_code'] == False]
        # Mismatched symptoms (both some symptoms and no symptoms selected) are removed
        filtered_data['symptom_any'] = filtered_data[['symptom_cough_any', 'symptom_new_continuous_cough',
       'symptom_runny_or_blocked_nose', 'symptom_shortness_of_breath',
       'symptom_sore_throat', 'symptom_abdominal_pain', 'symptom_diarrhoea',
       'symptom_fatigue', 'symptom_fever_high_temperature', 'symptom_headache',
       'symptom_change_to_sense_of_smell_or_taste', 'symptom_loss_of_taste',
       'symptom_other']].sum(axis=1) > 0
        self.mis_symptom_ids = list(filtered_data.loc[(filtered_data['symptom_any'] == 1) & (filtered_data['symptom_none'] == 1)]['participant_identifier'])
        filtered_data['symptom_mismatch'] = filtered_data['participant_identifier'].apply(lambda x: 1 if x in self.mis_symptom_ids else 0) == 1
        filtered_data = filtered_data.loc[filtered_data['symptom_mismatch'] == 0]
        filtered_data = filtered_data.loc[~((filtered_data['symptom_prefer_not_to_say']==1) & (filtered_data['symptom_none']==1))]
        filtered_data = filtered_data.drop('symptom_mismatch', axis = 1)
        # Replace 94+ age option with numeric replacement for purposes of plotting
        filtered_data['age'] = filtered_data['age'].apply(lambda x: 94 if x == '94+' else x)
        self.filtered_data = filtered_data
        # Define the dataset used for the train-test split, and the longitudinal test set
        self.longitudinal = filtered_data.loc[pd.to_datetime(filtered_data['submission_date']) + pd.to_timedelta(filtered_data['submission_hour'], unit='H') > '2021-11-29']
        self.train_test = filtered_data.loc[pd.to_datetime(filtered_data['submission_date']) + pd.to_timedelta(filtered_data['submission_hour'], unit='H') <= '2021-11-29']
        # Define colours for generating plots
        self.col_lst = ['#007C91', '#003B5C', '#582C83', '#1D57A5', '#8A1B61', '#E40046', '#00AB8E', '#00A5DF', '#84BD00', '#FF7F32', '#FFB81C', '#D5CB9F']
        
    
    def filter_missing_data(self, type='', save_figures=False, analyse=True):
        '''
        This method is used for analysing missing data and removing missing data from the either the entire dataset, or the train-test or longitudinal subsets of the data.
        
                Parameters:
                        type: Selects the data to perform function against. Either 'all', 'train_test', 'longitudinal'.
                        save_figures: Whether the figures plotted from analysis should be saved in the figures folder. Boolean True or False
                        analyse: Whether a json-string with some demographic frequencies should be printed. Boolean True or False
                        
                Returns: A pandas dataframe of the filtered data
        '''
        # Check that the type parameter has a valid entry
        if type not in ['all', 'train_test', 'longitudinal']:
            raise("type variable must be set to either 'train_test' or 'longitudinal'")
        if type == 'train_test':
            filtered_data = self.train_test
        elif type == 'longitudinal':
            filtered_data = self.longitudinal
        else:
            filtered_data = self.filtered_data           
        # Collect set with missing audio recordings
        size_cols = ['sentence_size', 'exhalation_size', 'cough_size', 'three_cough_size']
        for col in size_cols:
            filtered_data[col] = filtered_data[col].apply(lambda x: np.NaN if (x<=44 and str(x) != 'nan') else x)
        metadata_cols = ['height','weight','covid_test_result','covid_test_method', 'pseudonymised_local_authority_code','covid_test_processed_date','submission_delay']
        missing_audio = filtered_data.loc[filtered_data[size_cols].isna().sum(axis=1) >= 1]
        complete_data = filtered_data.loc[filtered_data[size_cols].isna().sum(axis=1) == 0]
        # Collect where meta-data is missing
        missing_meta = complete_data.loc[complete_data[metadata_cols].isna().sum(axis=1) >= 1]
        complete_data = complete_data.loc[complete_data[metadata_cols].isna().sum(axis=1) == 0]
        label_dict = {
            str(missing_meta): 'Missing Metadata',
            str(missing_audio): 'Missing Audio',
            str(complete_data): 'Complete Data'
        }
        # Set up JSON string and figures folder for analysis and plots
        if analyse or save_figures:
            analysis_dict = {
                'missing_audio': {'count': len(missing_audio) ,'gender': {}, 'covid_test_result': {}, 'symptoms': {}, 'language': {}},
                'missing_meta': {'count': len(missing_meta),'gender': {}, 'covid_test_result': {}, 'symptoms': {}, 'language': {}},
                'complete_data': {'count': len(complete_data),'gender': {}, 'covid_test_result': {}, 'symptoms': {}, 'language': {}}
            }
            dims = analysis_dict.keys()
            if not os.path.isdir('figures'):
                os.mkdir('figures')
            if not os.path.isdir('figures/missing_audio'):
                os.mkdir('figures/missing_audio')
            if not os.path.isdir('figures/missing_meta'):
                os.mkdir('figures/missing_meta')
            if not os.path.isdir('figures/complete_data'):
                os.mkdir('figures/complete_data')
                
            # In the analysis of each variable type, the code operates by first adding counts of each variable value to the json output, and then plotting each of the figures
            # Gender analysis
            gender_options = ['Male', 'Female', 'Unknown']
            for option in gender_options:
                analysis_dict['missing_meta']['gender'][option] = [str(x) for x in list(missing_meta['gender'])].count(option)
                analysis_dict['missing_audio']['gender'][option] = [str(x) for x in list(missing_audio['gender'])].count(option)
                analysis_dict['complete_data']['gender'][option] = [str(x) for x in list(complete_data['gender'])].count(option)
            if save_figures:
                for df_label in [[missing_audio, 'missing_audio'], [missing_meta, 'missing_meta'], [complete_data, 'complete_data']]:
                    df = df_label[0]
                    dim = df_label[1]
                    df = df_label[0]
                    crosstab_df = df[['gender', 'covid_test_result']]
                    crosstab_df.columns = ['Gender', 'Test Result']
                    crosstab_df['Gender'] = pd.Categorical(crosstab_df['Gender'], categories = ['Unknown', 'Male', 'Female'])
                    pd.crosstab(crosstab_df['Gender'], crosstab_df['Test Result'], margins=False, dropna=False).plot.barh(stacked=True, color = ['#E40046', '#00AB8E'], fontsize=15).legend(loc='lower right')
                    plt.savefig('figures/' + dim + '/gender.png', dpi=600, bbox_inches='tight')
            
            # Test result analysis
            test_result_options = ['nan', 'Negative', 'Positive']
            for option in test_result_options:
                analysis_dict['missing_meta']['covid_test_result'][option] = [str(x) for x in missing_meta['covid_test_result']].count(option)
                analysis_dict['missing_audio']['covid_test_result'][option] = [str(x) for x in missing_audio['covid_test_result']].count(option)
                analysis_dict['complete_data']['covid_test_result'][option] = [str(x) for x in complete_data['covid_test_result']].count(option)
            if save_figures:
                for dim in dims:
                    vals = [analysis_dict[dim]['covid_test_result'][x] for x in test_result_options]
                    test_result_df = pd.DataFrame([test_result_options, vals]).transpose()
                    test_result_df.columns = ['Test Result', 'Frequency']
                    test_result_df['Test Result'] = test_result_df['Test Result'].apply(lambda x:'Missing' if str(x).lower() == 'nan' else x)
                    fig, ax = plt.subplots()
                    plt.barh('Test Result', 'Frequency', data=test_result_df, color = ['#9E9E9E', '#00AB8E', '#E40046'])
                    ax.set_xlabel('Frequency', fontsize = 15)
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label.set_fontsize(15)
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(13)
                    plt.tight_layout()
                    plt.savefig('figures/' + dim + '/covid_test_result.png', dpi=600)
                    
            # Symptoms analysis
            symptom_columns = ['symptom_none', 'symptom_change_to_sense_of_smell_or_taste',
       'symptom_new_continuous_cough', 'symptom_abdominal_pain', 'symptom_cough_any', 'symptom_diarrhoea', 'symptom_fatigue', 'symptom_fever_high_temperature', 'symptom_headache', 'symptom_loss_of_taste', 'symptom_other', 'symptom_prefer_not_to_say', 'symptom_runny_or_blocked_nose', 'symptom_shortness_of_breath', 'symptom_sore_throat']
            for symptom in symptom_columns:
                analysis_dict['missing_meta']['symptoms'][symptom] = sum(missing_meta[symptom])
                analysis_dict['missing_audio']['symptoms'][symptom] = sum(missing_audio[symptom])
                analysis_dict['complete_data']['symptoms'][symptom] = sum(complete_data[symptom])
            if save_figures:
                # Get ordering of symptoms for plots in symps
                symps = pd.DataFrame(list(analysis_dict[dim]['symptoms'].items())).sort_values(by=1, ascending=True)
                symps[0] = symps[0].apply(lambda x: x.replace('symptom_', '').replace('_', ' ').capitalize())
                symps.columns = ['Symptoms', 'Frequency']
                new_symptom_columns = [x.replace('symptom_', '').replace('_', ' ').capitalize() for x in symptom_columns]
                for df_label in [[missing_audio, 'missing_audio'], [missing_meta, 'missing_meta'], [complete_data, 'complete_data']]:
                    df = df_label[0]
                    dim = df_label[1]
                    crosstab_df  = df[symptom_columns + ['covid_test_result']]
                    crosstab_df[symptom_columns] = crosstab_df[symptom_columns].astype('bool')
                    crosstab_df.columns = new_symptom_columns + ['Test Result']
                    cols = new_symptom_columns
                    crosstab_df.loc[:, cols] = crosstab_df.loc[:, cols].mul(cols)
                    crosstab_df['Symptoms'] = [[entry for entry in row if entry != ''] for row in crosstab_df[new_symptom_columns].values.tolist()]
                    crosstab_df = crosstab_df[['Symptoms', 'Test Result']].explode('Symptoms')
                    crosstab_df['Symptoms'] = pd.Categorical(crosstab_df.Symptoms, categories = list(symps['Symptoms']))
                    pd.crosstab(crosstab_df['Symptoms'], crosstab_df['Test Result'], margins=False, dropna=False).plot.barh(stacked=True, color = ['#E40046', '#00AB8E'], fontsize=12)
                    plt.savefig('figures/' + dim + '/symptomCount.png', dpi=600, bbox_inches='tight')
                    
            # language analysis
            language_options = ['English', 'Prefer not to say']
            for option in language_options:
                analysis_dict['missing_meta']['language'][option] = list(missing_meta['language']).count(option)
                analysis_dict['missing_audio']['language'][option] = list(missing_audio['language']).count(option)
                analysis_dict['complete_data']['language'][option] = list(complete_data['language']).count(option)
            analysis_dict['missing_meta']['language']['Other'] = analysis_dict['missing_meta']['count'] - sum(analysis_dict['missing_meta']['language'].values())
            analysis_dict['missing_audio']['language']['Other'] = analysis_dict['missing_audio']['count'] - sum(analysis_dict['missing_audio']['language'].values())
            analysis_dict['complete_data']['language']['Other'] = analysis_dict['complete_data']['count'] - sum(analysis_dict['complete_data']['language'].values())
            language_options = language_options + ['Other']
            if save_figures:
                for df_label in [[missing_audio, 'missing_audio'], [missing_meta, 'missing_meta'], [complete_data, 'complete_data']]:
                    df = df_label[0]
                    dim = df_label[1]
                    df = df_label[0]
                    crosstab_df = df[['language', 'covid_test_result']]
                    crosstab_df['language'] = crosstab_df['language'].apply(lambda x: x if x in ['Prefer not to say', 'English'] else 'Other')
                    crosstab_df.columns = ['Language', 'Test Result']
                    crosstab_df['Language'] = pd.Categorical(crosstab_df['Language'], categories = ['Prefer not to say', 'Other', 'English'])
                    pd.crosstab(crosstab_df['Language'], crosstab_df['Test Result'], margins=False, dropna=False).plot.barh(stacked=True, color = ['#E40046', '#00AB8E'], fontsize=12).legend(loc='lower right')
                    plt.savefig('figures/' + dim + '/language.png', dpi=600, bbox_inches='tight')
                    
            # Code for creating other figures used in Pigoli et. al. paper
            if save_figures:
                # Height by gender
                for df_label in [[missing_audio, 'missing_audio'], [missing_meta, 'missing_meta'], [complete_data, 'complete_data']]:
                    df = df_label[0]
                    dim = df_label[1]
                    height_df = df[['height', 'gender']]
                    height_df['height'] = height_df['height'].apply(lambda x: 90 if x == '<=90' else x)
                    height_df['height'] = height_df['height'].apply(lambda x: np.NaN if x == 'Prefer not to say' else x)
                    height_pnts = sum(height_df['height'].isna())
                    height_df['height'] = height_df['height'].astype(float)
                    height_df['gender'] = pd.Categorical(height_df['gender'], categories = ['Female', 'Male', 'Unknown'])
                    print(label_dict[str(df)] + ' Height Prefer Not To Say:' + str(height_pnts))
                    fig, ax = plt.subplots()
                    sns.boxplot(data =  height_df, x = 'height' , y = 'gender', palette=['#007C91', '#582C83', '#8A1B61'])
                    ax.set(ylabel=None)
                    ax.set_xlabel('Height (cm)', fontsize = 15)
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label.set_fontsize(15)
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(13)
                    plt.tight_layout()
                    plt.savefig('figures/' + dim + '/height_by_gender.png', dpi=600)
                    
                # Age by test result
                for df_label in [[missing_audio, 'missing_audio'], [missing_meta, 'missing_meta'], [complete_data, 'complete_data']]:
                    df = df_label[0]
                    dim = df_label[1]
                    age_df = df[['age', 'covid_test_result']]
                    age_df['age'] = age_df['age'].apply(lambda x: 94 if x == '94+' else x)
                    age_df['age'] = age_df['age'].astype(float)
                    fig, ax = plt.subplots()
                    sns.boxplot(data =  age_df, x = 'age' , y = 'covid_test_result', palette=['#00AB8E', '#E40046'])
                    ax.set(ylabel=None)
                    ax.set_xlabel('Age', fontsize = 15)
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label.set_fontsize(15)
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(13)
                    plt.tight_layout()
                    plt.savefig('figures/' + dim + '/age_by_test_result.png', dpi=600)
                    
                # Weight by gender
                for df_label in [[missing_audio, 'missing_audio'], [missing_meta, 'missing_meta'], [complete_data, 'complete_data']]:
                    df = df_label[0]
                    dim = df_label[1]
                    weight_df = df[['weight', 'gender']]
                    weight_df['weight'] = weight_df['weight'].apply(lambda x: np.NaN if x == 'Prefer not to say' else x)
                    weight_pnts = sum(weight_df['weight'].isna())
                    weight_df['gender'] = pd.Categorical(weight_df['gender'], categories = ['Female', 'Male', 'Unknown'])
                    print(label_dict[str(df)] + ' Weight Prefer Not To Say:' + str(weight_pnts))
                    fig, ax = plt.subplots()
                    sns.boxplot(data =  weight_df, x = 'weight' , y = 'gender', palette=['#007C91', '#582C83', '#8A1B61'])
                    ax.set(ylabel=None)
                    ax.set_xlabel('Weight (kg)', fontsize = 15)
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label.set_fontsize(15)
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(13)
                    plt.tight_layout()
                    plt.savefig('figures/' + dim + '/weight_by_gender.png', dpi=600)
                
                # Smoker Status by Covid result
                for df_label in [[missing_audio, 'missing_audio'], [missing_meta, 'missing_meta'], [complete_data, 'complete_data']]:
                    df = df_label[0]
                    dim = df_label[1]
                    crosstab_subset = df[['smoker_status', 'covid_test_result']]
                    crosstab_subset.columns = ['Smoker Status', 'Test Result']
                    pd.crosstab(crosstab_subset['Smoker Status'], crosstab_subset['Test Result'], margins=False).plot.barh(stacked=True, color = ['#E40046', '#00AB8E'], fontsize=15)
                    plt.savefig('figures/' + dim + '/smoker_status_by_test_result.png', dpi=600, bbox_inches='tight')
                    
                # Respiratory Conditions by Covid result
                for df_label in [[missing_audio, 'missing_audio'], [missing_meta, 'missing_meta'], [complete_data, 'complete_data']]:
                    df = df_label[0]
                    dim = df_label[1]
                    respiratory_plot = df[['respiratory_condition_none', 'respiratory_condition_asthma',
                           'respiratory_condition_copd_or_emphysema',
                           'respiratory_condition_other',
                           'respiratory_condition_prefer_not_to_say', 'covid_test_result']]

                    respiratory_plot.columns = [x.replace('respiratory_condition_', '').replace('_', ' ').capitalize() for x in respiratory_plot.columns[0:5]] + ['covid_test_result']
                    for col in respiratory_plot.columns[0:5]:
                        respiratory_plot[col] = respiratory_plot[col].apply(lambda x: col if x == 1 else '')
                    respiratory_plot['conditions'] = [[x for x in list(set(y)) if x != ''] for y in respiratory_plot.iloc[:, 0:5].values.tolist()]

                    crosstab_subset = respiratory_plot[['conditions', 'covid_test_result']].explode('conditions')
                    crosstab_subset.columns = ['Other Respiratory Conditions', 'Test Result']
                    pd.crosstab(crosstab_subset['Other Respiratory Conditions'], crosstab_subset['Test Result'], margins=False).plot.barh(stacked=True, color = ['#E40046', '#00AB8E'], fontsize=15)
                    plt.savefig('figures/' + dim + '/respiratory_conditions_by_test_result.png', dpi=600, bbox_inches='tight')
                    
                # Looking at height and weight by covid status for the complete data only
                for df_label in [[complete_data, 'complete_data']]:
                    df = df_label[0]
                    dim = df_label[1]
                    weight_df = df[['weight', 'covid_test_result']]
                    weight_df['weight'] = weight_df['weight'].apply(lambda x: np.NaN if x == 'Prefer not to say' else x)
                    weight_pnts = sum(weight_df['weight'].isna())
                    weight_df['covid_test_result'] = pd.Categorical(weight_df['covid_test_result'], categories = ['Positive', 'Negative'])
                    fig, ax = plt.subplots()
                    sns.boxplot(data =  weight_df, x = 'weight' , y = 'covid_test_result', palette=['#00AB8E', '#E40046'])
                    ax.set(ylabel=None)
                    ax.set_xlabel('Weight (kg)', fontsize = 15)
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label.set_fontsize(15)
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(13)
                    plt.tight_layout()
                    plt.savefig('figures/weight_by_test_result.png', dpi=600, bbox_inches='tight')
                
                for df_label in [[complete_data, 'complete_data']]:
                    df = df_label[0]
                    dim = df_label[1]
                    height_df = df[['height', 'covid_test_result']]
                    height_df['height'] = height_df['height'].apply(lambda x: 90 if x == '<=90' else x)
                    height_df['height'] = height_df['height'].apply(lambda x: np.NaN if x == 'Prefer not to say' else x)
                    height_pnts = sum(height_df['height'].isna())
                    height_df['height'] = height_df['height'].astype(float)
                    height_df['covid_test_result'] = pd.Categorical(height_df['covid_test_result'], categories = ['Positive', 'Negative'])
                    #print(label_dict[str(df)] + ' Height Prefer Not To Say:' + str(height_pnts))
                    fig, ax = plt.subplots()
                    sns.boxplot(data =  height_df, x = 'height' , y = 'covid_test_result', palette=['#00AB8E', '#E40046'])
                    ax.set(ylabel=None)
                    ax.set_xlabel('Height (cm)', fontsize = 15)
                    ax.set(xlim=[80, 240])
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label.set_fontsize(15)
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(13)
                    plt.tight_layout()
                    plt.savefig('figures/height_by_test_result.png', dpi=600, bbox_inches='tight')
                
                
        if analyse:
            print(json.dumps(analysis_dict, indent=2))
        return complete_data
    
    def train_test_split(self, asymptomatics_in_train=False, print_summary = False):
        '''
        This method is used to generate the train test splits used for models across the CIAB study
        
                Parameters:
                        asymptomatics_in_train: Whether any asymptomatics should be used for training. Boolean True or False
                        print_summary: Whether a summary of the test set construction should be printed. Boolean True or False
                        
                Returns: A dictionary object, keys are 'train', 'test', 'longitudinal', values are lists of participant identifiers
        '''
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        # Retrieve filtered train-test dataset
        train_test2 = self.filter_missing_data(type='train_test', analyse=False, save_figures=False)
        train_test2['covid_asymptomatic'] = (train_test2['symptom_none'] == 1) & (train_test2['covid_test_result'] == 'Positive')
        # Define asymptomatic submissions and calculate required sample sizes if asymptomatics are included in training set
        asymps = train_test2.loc[train_test2['covid_asymptomatic']]
        num_asymps = len(asymps['covid_asymptomatic'])
        target_asymps = int(np.ceil(num_asymps/2))
        # Retrieve filtered longitudinal test set
        longitudinal = self.filter_missing_data(type='longitudinal', analyse=False, save_figures=False)
        longitudinal_ids = longitudinal['participant_identifier']
        # Define no response entry for reproducibility of train-test splits used in analysis
        no_response = 'No Response' if asymptomatics_in_train else 'No response'
        # Select candidate test-set languages
        langs = [x for x in list(OrderedDict.fromkeys(train_test2['language'])) if x != 'English' and x != 'Prefer not to say']
        # Select candidate test-set ethnicities
        eths = [x for x in list(OrderedDict.fromkeys(train_test2['ethnicity'])) if x != 'British~ English~ Northern Irish~ Scottish~ or Welsh' and x != no_response and x != 'Another ethnic background']
        # Select candidate test-set geographies
        geographies = [x for x in list(OrderedDict.fromkeys(train_test2['pseudonymised_local_authority_code'])) if x not in ['LAD00262', 'LAD00257', 'LAD00272', 'LAD00048']]
        df_langs = len(langs)
        df_eths = len(eths)
        df_geogs = len(geographies)
        # Provide number of languages, ethnicities and local authorities to hold out for test set
        num_langs = 5
        num_eths = 5
        num_geogs = 4 
        # Randomly select languages, ethnicities and local authorities for test set
        test_langs = [langs[x] for x in random.sample(range(0, df_langs), num_langs)]
        print('Languages held back are: ' +  ', '.join(test_langs))
        test_eths = [eths[x] for x in random.sample(range(0, df_eths), num_eths)]
        print('Ethnicities held back are: ' +  ', '.join(test_eths))
        test_geogs = [geographies[x] for x in random.sample(range(0, df_geogs), num_geogs)]
        print('Local Authorities held back are: ' +  ', '.join(test_geogs))
        
        # a = language test set dataframe
        a = train_test2.loc[train_test2['language'].apply(lambda x: 1 if x in test_langs else 0) == 1]
        # b = ethnicity test set dataframe
        b = train_test2.loc[train_test2['ethnicity'].apply(lambda x: 1 if x in test_eths else 0) == 1]

        if not asymptomatics_in_train:
            # c = asymptomatic test set dataframe if all asymptomatics are included in the test set
            c = train_test2.loc[train_test2['covid_asymptomatic'] == 1]
        
        # Calculate median ages by gender
        m_age_med = np.quantile(train_test2.loc[train_test2['gender'] == 'Male']['age'], 0.5)
        f_age_med = np.quantile(train_test2.loc[train_test2['gender'] == 'Female']['age'], 0.5)
        
        # d = gender imbalanced dataframe for Covid-19 Positive cases for test set
        d = train_test2.loc[(((train_test2['gender'] == 'Male') & (train_test2['age'] > m_age_med)) | ((train_test2['gender'] == 'Female') & (train_test2['age'] > f_age_med))) & (train_test2['covid_test_result'] == 'Positive')]
        d = d.reset_index(drop=True)
        d_sample = random.sample(range(0, len(d)), round(len(d)*0.33))
        d = d.iloc[d_sample]
        
        # e = gender imbalanced dataframe for Covid-19 Negative cases for test set
        e = train_test2.loc[(((train_test2['gender'] == 'Male') & (train_test2['age'] < m_age_med)) | ((train_test2['gender'] == 'Female') & (train_test2['age'] < f_age_med))) & (train_test2['covid_test_result'] == 'Negative')]
        e = e.reset_index(drop=True)
        e_sample = random.sample(range(0, len(e)), round(len(e)*0.33))
        e = e.iloc[e_sample]
        
        # f = Test and Trace negatives or REACT-1 Positives for test set
        f = train_test2.loc[((train_test2['recruitment_source'] == 'REACT Study Round 13') & (train_test2['covid_test_result'] == 'Positive')) | ((train_test2['recruitment_source'] == 'REACT Study Round 14') & (train_test2['covid_test_result'] == 'Positive')) | ((train_test2['recruitment_source'] == 'Test and Trace') & (train_test2['covid_test_result'] == 'Negative'))]
        
        # g = local authorities dataframe for test set
        g = train_test2.loc[train_test2['pseudonymised_local_authority_code'].apply(lambda x: 1 if x in test_geogs else 0) == 1]
        # inclusion of strategically selected geographies in test set
        geog_pos = train_test2.loc[((train_test2['pseudonymised_local_authority_code'] == 'LAD00262') | (train_test2['pseudonymised_local_authority_code'] == 'LAD00257')) & (train_test2['covid_test_result'] == 'Positive')]
        geog_neg = train_test2.loc[((train_test2['pseudonymised_local_authority_code'] == 'LAD00048') | (train_test2['pseudonymised_local_authority_code'] == 'LAD00272')) & (train_test2['covid_test_result'] == 'Negative')]
        g = pd.concat([g, geog_pos, geog_neg])
        
        # Join all aforementioned dataframes to form initial test set construction
        test =  pd.concat([a, b, d, e, f, g]).drop_duplicates() if asymptomatics_in_train else pd.concat([a, b, c, d, e, f, g]).drop_duplicates()
        
        # Add a selection of asymptomatic cases to test set if some are being used for training
        if asymptomatics_in_train:
            current_asymps = test.loc[test['covid_asymptomatic']]['participant_identifier']
            remaining_asymps = target_asymps - len(current_asymps)
            possible_asymps = train_test2.loc[(train_test2['covid_asymptomatic'] == 1) & (train_test2['participant_identifier'].apply(lambda x: 1 if x in list(current_asymps) else 0) == 0)]['participant_identifier']
            testing_asymps = random.sample(list(possible_asymps), remaining_asymps)
            testing_asymps = train_test2.loc[train_test2['participant_identifier'].apply(lambda x:1 if x in testing_asymps else 0) == 1]

            test = pd.concat([test, testing_asymps])
        
        # Collect remaining submissions to be used for filling the test set for a 70-30 train-test split
        remaining_data = train_test2.loc[train_test2['participant_identifier'].apply(lambda x: 1 if x not in list(test['participant_identifier']) else 0) == 1].reset_index(drop=True)
        
        # Firstly randomly sample among submissions with different viral-loads to have this balanced in the test set
        viral_cat_counts = {}
        for x in list(OrderedDict.fromkeys(train_test2['covid_viral_load_category'])):
            if not pd.isna(x):
                viral_cat_counts[x] = list(test['covid_viral_load_category']).count(x)
        max_count = max(viral_cat_counts.values())

        for k in viral_cat_counts.keys():
            viral_cat_counts[k] = max_count - viral_cat_counts[k]

        for k in viral_cat_counts.keys():
            df_to_sample = remaining_data.loc[remaining_data['covid_viral_load_category'] == k].reset_index(drop=True)
            sub_samp = random.sample(range(0, len(df_to_sample)), viral_cat_counts[k])
            h = df_to_sample.iloc[sub_samp]
            test = pd.concat([test, h])

        test = test.drop_duplicates()

        viral_load_audio_sentences = list(train_test2.loc[~train_test2['covid_viral_load_category'].isna()]['participant_identifier'])
        
        # Calculate number of submissions remaining to be added to the test set
        remainder = round((len(train_test2) + 53) * 0.3) - len(test)
        if remainder <= 0:
            print('Test set exhausted')
        else:
            remaining_audio_sentences = [x for x in list(train_test2['participant_identifier']) if x not in list(test['participant_identifier']) and x not in list(viral_load_audio_sentences)]
            remaining_index = random.sample(range(0, len(remaining_audio_sentences)), remainder)
            extra_audio_sentences = [remaining_audio_sentences[x] for x in remaining_index]
            i = train_test2.loc[train_test2['participant_identifier'].apply(lambda x: 1 if x in extra_audio_sentences else 0) == 1]
            test = pd.concat([test, i])
            print(str(len(extra_audio_sentences)) + ' records added to test set at random')
        
        # Generate output dictionary
        test_ids = list(test['participant_identifier'])
        train_ids = [x for x in train_test2['participant_identifier'] if x not in test_ids]
        train_test_dict = {
            'train': train_ids,
            'test': test_ids,
            'longitudinal': list(longitudinal_ids)
        }
        print('Split successfully generated\n')
        # Check for duplicated records among the train or test sets
        if(len(train_test_dict['test']) == len(set(train_test_dict['test']))):
            print('No Test Set Duplicates')
        else:
            print('Duplicated IDs found in test set, check logic')
        if(len(train_test_dict['train']) == len(set(train_test_dict['train']))):
            print('No Train Set Duplicates')
        else:
            print('Duplicated IDs found in training set, check logic')
        # Print summary of test set construction
        if print_summary:
            print('\nTest set size: ' + str(len(train_test_dict['test'])))
            print('Train set size: ' + str(len(train_test_dict['train'])))
            print('Total: ' + str(len(train_test_dict['test']) + len(train_test_dict['train'])))
            print('\nTest Set Statistics:')
            print('Languages: ' + str(len(a['participant_identifier'])))
            print('Ethnicities: ' + str(len(b['participant_identifier'])))
            print('Asymptomatics: ' + str(sum(test['covid_asymptomatic'])))
            print('Age for positive cases: ' + str(len(d['participant_identifier'])))
            print('Age for negative cases: ' + str(len(e['participant_identifier'])))
            print('Positive geographies: ' + str(len(geog_pos['participant_identifier'])))
            print('Negative geographies: ' + str(len(geog_neg['participant_identifier'])))
            print('Other geographies: ' + str(len(g['participant_identifier']) - len(geog_pos['participant_identifier']) - len(geog_neg['participant_identifier'])))
            print('REACT positives: ' + str(len(train_test2.loc[((train_test2['recruitment_source'] == 'REACT Study Round 13') & (train_test2['covid_test_result'] == 'Positive')) | ((train_test2['recruitment_source'] == 'REACT Study Round 14') & (train_test2['covid_test_result'] == 'Positive'))]['participant_identifier'])))
            print('Test and Trace negatives: ' + str(len(train_test2.loc[((train_test2['recruitment_source'] == 'Test and Trace') & (train_test2['covid_test_result'] == 'Negative'))]['participant_identifier'])))
            for x in list(set(test['covid_viral_load_category'].dropna())):
                print(x + ': ' + str(list(test.loc[~test['covid_viral_load_category'].isna()]['covid_viral_load_category']).count(x)))
            print('END\n')
        
        return train_test_dict
    
     