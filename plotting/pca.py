'''
Script to visualise features generated from opensmile
author: Harry Coppock
Qs: harry.coppock@imperial.ac.uk
'''

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

from scipy import stats
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

def dim_red(df, list_features, method):
    '''
    Given the features and the labels performs PCA and projects onto the 2 principle
    components.
    '''
    # first scale the data
    X = df[list_features]
    y = df['test_result']
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    #perform dim red
    if method == 'pca':
        pca2 = PCA(n_components=2)
        princip_comp = pca2.fit_transform(X)
        principalDf = pd.DataFrame(data = princip_comp,
                            columns = ['principal component 1',
                                    'principal component 2'])
    elif method == 'tsne':
        pca2 = TSNE(n_components=2)
        princip_comp = pca2.fit_transform(X)
        principalDf = pd.DataFrame(data = princip_comp,
                            columns = ['t-SNE dimension 1',
                                    't-SNE dimension 2'])
    else:
        raise 'This is not a specified method for dim reduction'


    finalDf = pd.concat([principalDf, df], axis = 1)
    return finalDf

def plot_dim_red(finalDf, train_method, test_method, modality, hue, method):
    fig = plt.figure()
    if hue != 'age':
        g = sns.JointGrid(data=finalDf,
                    x="principal component 1" if method == 'pca' else "t-SNE dimension 1",
                    y="principal component 2" if method == 'pca' else "t-SNE dimension 2",
                    hue=hue)
        g.plot_joint(sns.scatterplot,
                    alpha=.5,
                    legend=True if hue != 'age' else False)

        sns.kdeplot(data=finalDf,
                        y="principal component 2" if method == 'pca' else "t-SNE dimension 2",
                        hue=hue,
                        linewidth=2,
                        ax=g.ax_marg_y, 
                        common_norm=False,
                        fill=True,
                        legend=False,
                        cut=0)
        
        sns.kdeplot(data=finalDf,
                        x="principal component 1" if method == 'pca' else "t-SNE dimension 1",
                        hue=hue,
                        linewidth=2, 
                        ax=g.ax_marg_x, 
                        common_norm=False, 
                        fill=True,
                        legend=False,
                        cut=0)
    else:
        cmap = sns.color_palette("YlOrBr", n_colors=1000, as_cmap=True)
        g = sns.scatterplot(
        data=finalDf,
        x="principal component 1" if method == 'pca' else "t-SNE dimension 1",
        y="principal component 2" if method == 'pca' else "t-SNE dimension 2",
        hue=hue,
        palette=cmap,
        legend=False,
        )
        divider = make_axes_locatable(g)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        #fig.colorbar(im1, cax=cax, orientation='vertical')
        points = plt.scatter([], [], c=[], vmin=min(finalDf['age'].tolist()), vmax=max(finalDf['age'].tolist()), cmap=cmap)
        plt.colorbar(points, cax=cax)
        #ax.subplots_adjust(right=.92)
        #cax = g.add_axes([.94, .25, .02, .6])
    plt.savefig(f'.figs/ssast/{train_method}/{test_method}/{method}_{modality}_{hue}.png', bbox_inches='tight')
    return g

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

if __name__ == '__main__':

    # f'/home/ec2-user/SageMaker/jbc-cough-in-a-box/SvmBaseline/features/opensmile2/{modality}'   
    #fig, axs = plt.subplots(6,3, figsize=(20,20))
    #fig = plt.figure(figsize=(15,15))
    #axs = gridspec.GridSpec(6, 3)
    POSSIBLE_train = ['standard-train', 'naive', 'big', 'matched-train']
    POSSIBLE_test = ['standard_test', 'matched_test', 'long_test']#, 'analysis_train']
    POSSIBLE_MODALITIES = [#'audio_cough_url',
                        #'three_cough',
                        'sentence']
                        #'audio_ha_sound_url']
                          
    POSSIBLE_CONTROLS = [
                "age",
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
    for train_method in POSSIBLE_train:
        for row, test_method in enumerate(POSSIBLE_test):
            for method in POSSIBLE_METHODS:        
                for modality in tqdm(POSSIBLE_MODALITIES):
                    if train_method == 'naive' and (test_method == 'matched_test' or test_method == 'long_test'):
                        continue
                    if train_method == 'big' and test_method == 'long_test':
                        continue
                    path = f'/home/ec2-user/SageMaker/jbc-cough-in-a-box/ssast/src/finetune/ciab/exp/test01-ciab_{modality}-f128-128-t1-2-b20-lr1e-4-ft_avgtok-base-unknown-SSAST-Base-Frame-400-1x-noiseTrue-{train_method}-final/fold1/{test_method}_pca_projections.csv'
                    if not os.path.exists(f'.figs/ssast/{train_method}/{test_method}'):
                        os.makedirs(f'.figs/ssast/{train_method}/{test_method}')
                    data_df, list_features = load_features(path)    
                    df = dim_red(data_df, list_features, method)
                    for column, control in enumerate(POSSIBLE_CONTROLS):
                        plot_dim_red(df, train_method, test_method, modality, control, method)#, ax=axs[column, row], fig=fig)


