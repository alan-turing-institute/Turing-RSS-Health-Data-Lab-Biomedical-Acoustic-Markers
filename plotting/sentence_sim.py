import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

sys.path.append('../')

from utils.dataset_stats import DatasetStats

def plot_ranking(meta, modality):
    #just for colouring which ones we checked
    sorted_meta = meta.sort_values(f'Sentence Outlier Scores ({modality})')
    sorted_meta = sorted_meta.drop_duplicates(subset=f'Participants said: ({modality})')
    checked = sorted_meta.iloc[:1000,:]
    meta['checked'] = meta[f'Participants said: ({modality})'].apply(lambda x: 'checked' if x in checked[f'Participants said: ({modality})'].tolist() else 'not checked')
    meta = meta[meta[f'Participants said: ({modality})'] != 'Not possible to load audio']
    min_score, max_score = min(meta[f'normalised_scores_{modality}']), max(meta[f'normalised_scores_{modality}'])
    h, m, l, v = extract_examples(meta, modality, min_score, max_score)
    v = v[:20]
    h = h[:20]
    l = l[:20]
    m = m[:20]
    fig, ax = plt.subplots(1,1)
    sns.histplot(data=meta, x=f'normalised_scores_{modality}', log_scale=(False, True), hue='checked', ax=ax)
    ax.annotate(h[2:-2],
            xy=(0,1000),
            xycoords='data',
            xytext=(-10,60),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->"))

    ax.annotate(m[2:-2],
            xy=(max_score-(max_score*0.75),200),
            xycoords='data',
            xytext=(0,40),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->"))
    ax.annotate(l[2:-2],
            xy=(max_score-(max_score*0.5),20),
            xycoords='data',
            xytext=(-80,60),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->"),
            wrap=True)
    ax.annotate(v[2:20],
            xy=(max_score,1),
            xycoords='data',
            xytext=(-40,40),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->"),
            wrap=True)
    plt.savefig(f'figs/scores_hist{modality}.png', bbox_inches='tight')

def extract_examples(meta, modality, min_score, max_score):
    high_sim = meta[meta[f'normalised_scores_{modality}'] == 0.0].iloc[0]
    med_sim = meta[meta[f'normalised_scores_{modality}'].between(max_score-(max_score*0.75), max_score-(max_score*0.5))].iloc[0]
    low_sim = meta[meta[f'normalised_scores_{modality}'].between(max_score-(max_score*0.5), max_score-(max_score*0.25))].iloc[1]
    very_low_sim = meta[meta[f'normalised_scores_{modality}'].between(max_score-(max_score*0.25),max_score)].iloc[0]

    return high_sim[f'Participants said: ({modality})'], med_sim[f'Participants said: ({modality})'], low_sim[f'Participants said: ({modality})'], very_low_sim[f'Participants said: ({modality})']




if __name__ == '__main__':

    meta = pd.read_csv('../utils/meta_data_v5_a.csv') 
    print(meta.columns)
    POSSIBLE_MODALITIES = [#'audio_sentence_url',
                   'audio_ha_sound_url',
                   'audio_cough_url',
                   'audio_three_cough_url']
    for modality in POSSIBLE_MODALITIES:
        meta[f'normalised_scores_{modality}'] = meta[f'Sentence Outlier Scores ({modality})'] - meta[f'Sentence Outlier Scores ({modality})'].max()
        meta[f'normalised_scores_{modality}'] = meta[f'normalised_scores_{modality}'].abs()
        plot_ranking(meta, modality)

