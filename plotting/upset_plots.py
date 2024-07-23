'''
function to create the upset plot
author: harry.coppock@imperial.ac.uk
'''
import sys, math
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
#     study_data['log_covid_viral_load'] = study_data['covid_viral_load'].apply(lambda x: math.log(x, 10) if x > 0 else 0)
    if 'symptoms' not in study_data.columns:
        study_data['symptoms'] = study_data.apply(lambda row: [re_word(col) for col in ['symptom_none','symptom_cough_any','symptom_new_continuous_cough','symptom_runny_or_blocked_nose','symptom_shortness_of_breath','symptom_sore_throat','symptom_abdominal_pain','symptom_diarrhoea','symptom_fatigue','symptom_fever_high_temperature','symptom_headache','symptom_change_to_sense_of_smell_or_taste','symptom_loss_of_taste','symptom_other','symptom_prefer_not_to_say'] if row[col] == 1], axis=1)

    covid_by_symptoms = from_memberships(
        #    study_data[study_data['covid_test_result'] == covid].symptoms.apply(lambda x: ','.join(x)).apply(lambda x: x.replace('A change to sense of smell or taste','$\Delta$ sense of smell or taste').replace('(a high temperature)', '')).apply(lambda x: x.replace('Other symptom(s) new to you in the last 2 weeks','Other new symptoms(s)')).apply(lambda x: x.replace(';','')).str.split(','),
           study_data[study_data['covid_test_result'] == covid]['symptoms'],
           data=study_data[study_data['covid_test_result'] == covid])
    upset = UpSet(covid_by_symptoms,
            min_subset_size=148 if covid == 'Positive' else 122,
            show_counts=True,
            sort_by='cardinality',
            facecolor= 'darkred' if covid == 'Positive' else 'mediumblue')
    #upset.add_stacked_bars(by='gender')
    if covid == 'Positive':
        upset.add_catplot(value='covid_ct_value', kind='violin', color='grey', edgecolor='black')
    #upset.style_subsets(present='Positive', label='Covid Positive', facecolor='red')
    #upset.style_subsets(present='Negative', label='Covid Negative', facecolor='green')
    upset.plot()
    positive_plot = plt.gcf()
    ax = plt.gca()
    trans = mtransforms.ScaledTranslation(-200/72, 0/72, positive_plot.dpi_scale_trans)
#     ax.text(
#             0.0,
#             1.0,
#             'e) Positive Participants' if covid == 'Positive' else 'f) Negative Participants',
#             transform=ax.transAxes + trans,
#             fontsize='x-large', verticalalignment='top', fontfamily='serif',
#             bbox=dict(facecolor='1', edgecolor='none', pad=3.0, alpha=0))
    positive_plot.savefig(f'../figs/{covid}_upset.pdf', bbox_inches='tight')
    plt.close()

def re_word(text):
    dic = {
        'symptom_': '',
        'none': 'No symptoms',
        'prefer_not_to_say': 'Prefer not to say',
        'abdominal_pain': 'Abdominal pain',
        'diarrhoea': 'Diarrhoea',
        'other' : 'Other new symotom(s)',
        'new_continuous_cough': 'New continuous cough',
        'loss_of_taste': 'Loss of taste',
        'shortness_of_breath': 'Shortness of breath',
        'fever_high_temperature': 'Fever/high temperature',
        'change_to_sense_of_smell_or_taste': '$\Delta$ sense of smell or taste',
        'sore_throat': 'Sore throat',
        'headache': 'Headache',
        'runny_or_blocked_nose': 'Runny or blocked nose',
        'fatigue': 'Fatigue',
        'cough_any': 'Cough',

    }
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

if __name__ == '__main__':
    upset('Positive')
    upset('Negative')
