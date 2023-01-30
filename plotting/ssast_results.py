import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (classification_report,
                            confusion_matrix,
                            recall_score,
                            precision_score,
                            make_scorer,
                            RocCurveDisplay,
                            plot_roc_curve,
                            PrecisionRecallDisplay,
                            roc_curve,
                            auc,
                            roc_auc_score,
                            precision_recall_curve,
                            average_precision_score)
POSSIBLE_MODALITIES = ['audio_ha_sound_url',
                        'audio_cough_url',
                        'audio_three_cough_url',
                        'audio_sentence_url']

def roc_auc(fpr, tpr, train_method, test_method, modality):
    '''
    produce roc curve and savefig
    '''
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(modality)
    plt.legend(loc="lower right") 
    plt.savefig(f'figs/ssast/final/{modality}_{train_method}_{test_method}roccurve.png',
                bbox_inches='tight') 
    plt.close()

def pr_auc(rec, pr, train_method, test_method, modality):
    '''
    produce pr curve and savefig
    '''
    pr_auc = auc(rec, pr)
    plt.figure()
    lw = 2
    plt.plot(
        rec,
        pr,
        color="darkorange",
        lw=lw,
        label="PR curve (area = %0.2f)" % pr_auc,
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.title(modality)
    plt.savefig(f'figs/ssast/final/{modality}_{train_method}_{test_method}prcurve.png',
                bbox_inches='tight') 

    plt.close()

def load_metrics(path):
    with open(path) as f:
        metrics = json.load(f)
    return metrics

def create_x_name(x):
    if 'naive' in x:
        return 'naive_method'
    elif 'matched' in x:
        return 'matched_train'
    elif 'standard' in x:
        return 'standard_train'

def create_dataframe(metrics, train_method):
    
    metrics.pop('val')
    metrics.pop('analysis_train_dataset')
    metrics = pd.DataFrame.from_dict(metrics, orient='index')
    metrics.reset_index(inplace=True)
    metrics = metrics.rename(columns={'index':'test_type'})
    metrics['name'] = metrics['test_type'] + train_method
    metrics['x_name'] = 'Test set: ' +  metrics['test_type'] + '\n Train set: ' + train_method
    metrics['x_name'] = metrics['x_name'].apply(lambda x: 'naive_method' if 'naive' in x else x)
    return metrics

def plot_summary_bar(df, modality):
    x = np.arange(len(df))  # the label locations
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots(figsize=(10,8))
    rects1 = ax.bar(x - width,
            df['uar'].tolist(),
            width,
            label='UAR',
            color='#007C91')
    rects2 = ax.bar(x,
            df['AP'],
            width,
            label='PR AUC',
            color='#003B5C')
    rects3 = ax.bar(x + width,
            df['auc'],
            width,
            label='ROC AUC',
            color='#582C83')


    #  Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_ylim(bottom=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(df['x_name'].tolist())
    ax.tick_params(axis='x', rotation=80)
    ax.legend()
    ax.grid(True)
    ax.set_title(modality)
    plt.savefig(f'figs/ssast/final/summary_{modality}.png',
    bbox_inches='tight'
    )


if __name__ == '__main__':
    POSSIBLE_train = ['standard-train', 'naive', 'matched-train', 'original-train', 'original-matched-train']
    POSSIBLE_test = ['test', 'matched_test', 'long_test', 'matched_long_test']
    POSSIBLE_MODALITIES = ['cough',
                        'three_cough',
                        'sentence',
                        'ha_sound']
    for modality in POSSIBLE_MODALITIES: 
        for i, train_method in enumerate(POSSIBLE_train):
            path = f'/workspace/ssast_ciab/src/finetune/ciab/exp/final/ciab_{modality}-f128-128-t1-2-b20-lr1e-4-ft_avgtok-base-unknown-SSAST-Base-Frame-400-1x-noiseTrue-{train_method}/fold1/metrics.json'

            metrics = load_metrics(path)
            if i == 0:
                df = create_dataframe(metrics, train_method)
            else:
                temp = create_dataframe(metrics, train_method)
                df = pd.concat([df, temp], axis=0)
            for test_method in POSSIBLE_test:
                if train_method == 'naive' and test_method in ['matched_test', 'long_test', 'matched_long_test']:
                    continue

                prec = df[df['name'] == test_method + train_method].precisions.tolist()
                rec = df[df['name'] == test_method + train_method].recalls.tolist()
                fpr = df[df['name'] == test_method + train_method].fpr.tolist()
                #print(np.ones_like(np.array(df[df['test_type'] == test_method].fnr.tolist())))
                #print(np.array(df[df['test_type'] == test_method].fnr.tolist()))
                tpr = np.ones_like(np.array(df[df['name'] == test_method + train_method].fnr.tolist())) - np.array(df[df['name'] == test_method + train_method].fnr.tolist())
                #print( np.array(df[df['test_type'] == test_method].fnr.tolist()))
                pr_auc(np.array(rec[0]), np.array(prec[0]), train_method, test_method, modality)
                roc_auc(np.array(fpr[0]), np.array(tpr[0]), train_method, test_method, modality)
        plot_summary_bar(df, modality)
