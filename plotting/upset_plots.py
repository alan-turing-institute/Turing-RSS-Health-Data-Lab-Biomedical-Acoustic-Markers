'''
function to create the upset plot
author: harry.coppock@imperial.ac.uk
'''
import sys 
sys.path.append('../')
import pandas
from upsetplot import UpSet, from_contents, from_memberships
from utils.dataset_stats import DatasetStats
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

def upset(covid):
    plt.rc('xtick',labelsize=12)
    plt.rc('ytick',labelsize=12)
    font={'size': 10}
    plt.rc('font', **font)
    dataset = DatasetStats()
    study_data = dataset.meta_data
    covid_by_symptoms = from_memberships(
           study_data[study_data['test_result'] == covid].symptoms.apply(lambda x: ','.join(x)).apply(lambda x: x.replace('A change to sense of smell or taste','$\Delta$ sense of smell or taste').replace('(a high temperature)', '')).apply(lambda x: x.replace('Other symptom(s) new to you in the last 2 weeks','Other new symptoms(s)')).apply(lambda x: x.replace(';','')).str.split(','),
           data=study_data[study_data['test_result'] == covid])
    upset = UpSet(covid_by_symptoms,
            min_subset_size=148 if covid == 'Positive' else 122,
            show_counts=True,
            sort_by='cardinality',
            facecolor= 'darkred' if covid == 'Positive' else 'mediumblue')
    #upset.add_stacked_bars(by='gender')
    #upset.add_catplot(value='ct_value', kind='strip', color='grey', edgecolor='black')
    #upset.style_subsets(present='Positive', label='Covid Positive', facecolor='red')
    #upset.style_subsets(present='Negative', label='Covid Negative', facecolor='green')
    upset.plot()
    positive_plot = plt.gcf()
    ax = plt.gca()
    trans = mtransforms.ScaledTranslation(-200/72, 0/72, positive_plot.dpi_scale_trans)
    ax.text(
            0.0,
            1.0,
            'e) Positive Participants' if covid == 'Positive' else 'f) Negative Participants',
            transform=ax.transAxes + trans,
            fontsize='x-large', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0, alpha=0))
    positive_plot.savefig(f'figs/{covid}_upset.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    upset('Positive')
    upset('Negative')
