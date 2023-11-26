'''
File visulise the geographical distribution of the collected dataset
author: Harry Coppck
Qs to harry.coppock@imperial.ac.uk
'''

import pandas as pd
import json 
import numpy as np
import geopandas
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import boto3
import pandas
import io
from botocore import UNSIGNED
from botocore.config import Config
import yaml
import sys
sys.path.append('../')
from utils.dataset_stats import DatasetStats



def get_file(path):
    return io.BytesIO(bucket.Object(path).get()['Body'].read())


def plot_geo(data, covid, ax, percent=True):
    '''
    data: geo data
    covid: str {Positive, Negative, Difference}
    ax: axis to plot the figure
    '''
    dataset = DatasetStats()
    study_data = dataset.meta_data
    if covid == 'all':
        total_county_counts = study_data['local_authority_code'].value_counts().rename_axis('LAD19CD').reset_index(name='counts')

        vmin, vmax, = total_county_counts.counts.min(), total_county_counts.counts.max() 

    elif covid != 'Positive - Negative' and covid != 'Negative - Positive':
        total_county_counts = study_data[
                study_data['test_result']==covid
                    ]['local_authority_code'].value_counts().rename_axis('LAD19CD').reset_index(name='counts')
        vmin, vmax, = total_county_counts.counts.min(), total_county_counts.counts.max() 
    else:
        positive_counts = study_data[
                study_data['test_result']=='Positive'
                    ]['local_authority_code'].value_counts().rename_axis('LAD19CD').reset_index(name='positive counts')
        negative_counts = study_data[
                study_data['test_result']=='Negative'
                    ]['local_authority_code'].value_counts().rename_axis('LAD19CD').reset_index(name='negative counts')
        if percent == True:
            print(positive_counts)
            positive_counts['positive counts'] = positive_counts['positive counts'] / positive_counts['positive counts'].sum() * 100
            negative_counts['negative counts'] = negative_counts['negative counts'] / negative_counts['negative counts'].sum() * 100

        total_county_counts = pd.merge(positive_counts, negative_counts, on='LAD19CD', how='outer')
        total_county_counts = total_county_counts.fillna(0)
        if covid == 'Positive - Negative':
            total_county_counts['counts'] = total_county_counts['positive counts'] - total_county_counts['negative counts']
        else:
            total_county_counts['counts'] = total_county_counts['negative counts'] - total_county_counts['positive counts']
        print(total_county_counts['counts'])
        vmin = min(total_county_counts.counts.min(), -total_county_counts.counts.max())
        vmax = max(-total_county_counts.counts.min(), total_county_counts.counts.max())
        vcenter = 0
        norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
        cbar = plt.cm.ScalarMappable(norm=norm, cmap='RdBu')
    num_county_before = len(total_county_counts)
    combined = pd.merge(data, total_county_counts,  on=['LAD19CD'], how='outer')
    print('final sum', combined['counts'].sum())
    if covid == 'Positive - Negative' or covid == 'Negative - Positive':
        cmap = 'RdBu'
    elif covid == 'Positive':
        cmap = 'Reds'
    else:
        cmap = 'PuBu'
    combined.plot(ax=ax,
                  column='counts',
                  legend=False,
                  vmin=vmin if not (covid == 'Positive - Negative' or covid == 'Negative - Positive') else None,
                  vmax=vmax if not (covid == 'Positive - Negative' or covid == 'Negative - Positive') else None,
                  missing_kwds={'color': 'lightgrey',
                                'label':'No submissions'},
                  cmap=cmap,
                  norm=norm if (covid == 'Positive - Negative' or covid == 'Negative - Positive') else None)
    print('total_num_cases', len(study_data[study_data['test_result']==covid]))
    num_count_after = len(combined[combined['counts'] > 0])
    print('lost counties', num_count_after - num_county_before)
    ax.axis('off')
    #title = covid if covid != 'all' else 'Study Participant Locations'
    #ax.set_title(title)
    if covid == 'Positive - Negative' or covid == 'Negative - Positive':
        return cbar

if __name__ == '__main__':
    
    #load the uk county boundaries
    url = 'https://opendata.arcgis.com/api/v3/datasets/83f458a118604169b599000411f364bf_0/downloads/data?format=shp&spatialRefId=27700'
    data = geopandas.read_file(url)

    #load the study data

    try:
        with open('config.yml', 'r')as conf:
            PATHS = yaml.safe_load(conf)
    except FileNotFoundError as err:
        raise ValueError(f'You need to specify your local paths to the data and meta data: {err}')
    bucket_name = PATHS['meta_bucket']
    s3_resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED), region_name='eu-west-2')
    bucket = s3_resource.Bucket(bucket_name)


    fig, axs = plt.subplots(1,2, figsize=(15,10))

    plot_geo(data, 'Positive', axs[0])
    plot_geo(data, 'Negative', axs[1])
    im = plt.gca().get_children()[0]
    cax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
    fig.colorbar(im, cax=cax)
    plt.savefig(f'map_uk_all.pdf')


    fig, ax = plt.subplots(1,1, figsize=(7.5, 10))
    cbar = plot_geo(data, 'Positive - Negative', ax)
    cax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
    fig.colorbar(cbar, cax=cax)
    plt.savefig(f'map_uk_difference.pdf')
    
