import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
POSSIBLE_MODALITIES = ['audio_ha_sound_url',
                        'audio_cough_url',
                        'audio_three_cough_url',
                        'audio_sentence_url']


def load_metrics(path):
    '''
    iterate through all .json files in a directory path
    '''
    metrics = {} 
    for root, dirs, files in os.walk(path):
        for file in files:
            if '.json' in file and 'ipynb' not in root:
                with open(os.path.join(root, file), 'r') as f:
                    metrics[root+ file] = json.load(f)
    return metrics

def create_x_name(x):
    if 'naive' in x:
        return 'naive_method'
    elif 'matched' in x:
        return 'matched_train'
    elif 'standard' in x:
        return 'standard_train'

def create_dataframe(metrics):
    
    for i, (key, item) in enumerate(metrics.items()):
        item.pop('params')
        item.pop('dev')
        if i == 0:
            metrics = pd.DataFrame.from_dict(item, orient='index')
            metrics['mode'] = key
        else:
            temp = pd.DataFrame.from_dict(item, orient='index')
            temp['mode'] = key
            metrics = pd.concat([metrics, temp], axis=0)
    metrics.reset_index(inplace=True)
    metrics = metrics.rename(columns={'index':'test_type'})
    metrics['x_name'] = 'Test set: ' +  metrics['test_type'] + '\n Train set: ' + metrics['mode'].apply(create_x_name)
    metrics['x_name'] = metrics['x_name'].apply(lambda x: 'naive_method' if 'naive' in x else x)
    return metrics

def plot_summary_bar(df, modality):
    x = np.arange(len(df))  # the label locations
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width,
            df['uar_mean'].tolist(),
            width,
            label='UAR',
            color='#007C91',
            yerr=[df['uar_mean'] - df['uar_ci_low'], df['uar_ci_high'] - df['uar_mean']],
            capsize=3)
    rects2 = ax.bar(x,
            df['pr_auc_mean'],
            width,
            label='PR AUC',
            color='#003B5C',
            yerr=[df['pr_auc_mean'] - df['pr_auc_ci_low'], df['pr_auc_ci_high'] - df['pr_auc_mean']],
            capsize=3)
    rects3 = ax.bar(x + width,
            df['roc_auc_mean'],
            width,
            label='ROC AUC',
            color='#582C83',
            yerr=[df['roc_auc_mean'] - dfk['roc_auc_ci_low'], df['roc_auc_ci_high'] - df['roc_auc_mean']],
            capsize=3)


    #  Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_ylim(bottom=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(df['x_name'].tolist())
    ax.tick_params(axis='x', rotation=80)
    ax.legend()
    ax.grid(True)
    plt.savefig(f'figs/summary_{modality}.png',
    bbox_inches='tight'
    )


if __name__ == '__main__':

    metrics = load_metrics('/home/ec2-user/SageMaker/jbc-cough-in-a-box/SvmBaseline/results_feb_2022/')
    metrics = create_dataframe(metrics)
    plot_summary_bar(metrics[metrics['mode'].apply(lambda x: 'three_cough' in x)], 'three_cough')
    plot_summary_bar(metrics[metrics['mode'].apply(lambda x: ('sentence' not in x) and ('three' not in x) and ('ha_sound' not in x) and ('cough' in x))], 'cough')
    plot_summary_bar(metrics[metrics['mode'].apply(lambda x: 'sentence' in x)], 'sentence')
    plot_summary_bar(metrics[metrics['mode'].apply(lambda x: 'ha_sound' in x)], 'ha_sound')
