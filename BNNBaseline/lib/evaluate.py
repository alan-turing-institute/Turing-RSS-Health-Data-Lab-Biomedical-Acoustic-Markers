import sys
import os

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import metrics
import config
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, accuracy_score

from collections import defaultdict

import matplotlib.pyplot as plt


from load_feat import load_data
import pickle
import models



def model_predict(model, test_loader, y_test, criterion=nn.BCELoss(), device=None, n_samples=1):
    ''' Predict for a model object given a PyTorch dataloader, loss criterion, device, and number of MC dropout samples.
        Inputs: `model`: PyTorch model class object
                `test_loader`: Tensor Dataset constructed with DataLoader
                `y_test`: label array to initialise BNN prediction array, shape [n_samples, len(y_test), n_classes].
                `criterion`: Loss function, by default, BCELoss(). Note that some criterions may need two units as the
                             final hidden layer output (e.g. CrossEntropyLoss).
                `device`: (optional), device for prediction, selects a single cuda GPU if available or CPU
                `n_samples`: defaults to 1 in case the network is deterministic at test time. As this is a BNN,
                             we approximate the posterior by sampling from the network at test time, with n_samples.
                             
        Outputs: `all_y`: Outputs ground truth y labels.
                 `y_preds_all`: concatenation of all outputs over batches, given in shape [n_samples, len(y), n_classes]
    '''
    
    if device is None:
        torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    y_preds_all = np.zeros([n_samples, len(y_test), 2])

    for n in range(n_samples):
        all_y_pred = []
        all_y = []
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x).squeeze()
            all_y.append(y.cpu().detach())

            all_y_pred.append(y_pred.cpu().detach())

            del x
            del y
            del y_pred

        all_y_pred = torch.cat(all_y_pred)
        all_y = torch.cat(all_y)

        y_preds_all[n,:,1] = np.array(all_y_pred)
        y_preds_all[n,:,0] = 1-np.array(all_y_pred) # Enter class probabilities per dropout sample
    
    
    return all_y, y_preds_all

def evaluate_metrics(labels, predictions):
    ''' Evaluate standard metrics with scikit-learn built in libraries.
        Inputs: `labels`: y labels
                `predictions`: model predictions, 1-D array with probabilities
        Outputs: `results`: dict of results of UAR, ROC-AUC and PR-AUC.
    '''
    results = {}
    # dicts with scikit-learn default metrics
    results["UAR"] = recall_score(labels, np.round(predictions), average='macro')
    results["ROC-AUC"] = roc_auc_score(labels, predictions)
    results["PR-AUC"] = average_precision_score(labels, predictions, average='macro')
    return results



def evaluate_model(audio_modality, train_type, test_type, load_part_only=False, n_samples=1):  # Make as function of data split also
    ''' Evaluate model on pickled data, according to a supplied audio_modality, and experiment
        type as denoted by train_type and test_type. This function assumes that the signal is broken
        into windows in a flattened array. We output the model evaluation over individual participant ID,
        using the original label list from the metadata. Predictions over windows are mean-averaged to 
        produce a single prediction per participant, which is then evaluated in downstream metrics.
        
        Inputs: `audio_modality`: Audio modality as referenced from dataframe, by default in:
                                  ['sentence_url', 'exhalation_url_url','cough_url', 'three_cough_url']
                `train_type`: Used to denote which pickled feature outputs to load. Logic also checks
                              for this string when loading corresponding model for evaluation.
                `test_type`: Denote which pickled feature outputs to load.
                `load_part_only`: Load a single pickle part for debugging purposes, by default False.
                `n_samples`: Number of samples to predict over for the BNN. For deterministic networks,
                             set to 1.
               
        Outputs: `results`: dict of results containing UAR, ROC-AUC and PR-AUC, calculated by evaluate_metrics().
                 `aggregate_results`: aggregated predictions per participant id
                 `aggregate_labels`: aggregate labels per participant id (same dimension as original list from metadata)
                 `G_X` predictive entropy
                 `U_X: mutual information
    '''
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else torch.device("cpu"))
    print(f'Evaluation on {device}')

    model = models.ResnetDropoutFull()  # Instantiate model from models.py. Add your own there if you wish


    model = model.to(device)

    x_test, y_test_list = load_data(audio_modality, test_type, load_part_only = load_part_only, stacked=False)
        # Now predict over windows first
    x_test = torch.tensor(np.vstack(x_test)).float()
    y_test = torch.tensor(np.hstack(y_test_list)).unsqueeze(1).float()
    x_test = x_test.unsqueeze(1).repeat(1,3,1,1)  # Repeat over 3 channels to match ResNet pre-trained assumptions
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False) # Larger batch size for evaluation
        
    output_filename = 'outputs_' + train_type + '_' + audio_modality + '_resnet_'
    for filename in os.listdir(os.path.join(os.path.pardir, 'models', train_type, audio_modality)):
        if output_filename in filename:
            with open(os.path.join(os.path.pardir, 'models', train_type, audio_modality, filename), "rb") as input_file:
                output = pickle.load(input_file)
                e = output['best_epoch']
                print('Best epoch', e)

    for checkpoint_name in os.listdir(os.path.join(os.path.pardir, 'models', train_type, audio_modality)):
        if audio_modality + '_e' + str(e) + '_resnet' in checkpoint_name:
            print('Predicting for:', checkpoint_name, audio_modality)
            checkpoint = model.load_state_dict(torch.load(os.path.join(os.path.pardir, 'models', train_type,
                                                                       audio_modality, checkpoint_name)))
            

            all_y, all_y_pred = model_predict(model, test_loader, y_test, device=device, n_samples=n_samples)
            

            prev_length = 0

            aggregate_predictions = []
            aggregate_labels = []

            for participant_label in y_test_list:
                length = len(participant_label)

                prediction = all_y_pred[:,prev_length:prev_length+length,1]
                if length: # Remove NaNs/missing audio < length of window for prediction
                    aggregate_predictions.append(np.mean(prediction, axis=1))
                    aggregate_labels.append(int(participant_label[0]))
                prev_length += length
            results = evaluate_metrics(np.array(aggregate_labels), np.array(np.mean(aggregate_predictions, axis=1)))
            print(results) 
            # Initialise results in format compatible with MI, PE calculation
            aggregate_results = np.zeros([np.shape(aggregate_predictions)[1], np.shape(aggregate_predictions)[0], 2])
            aggregate_results[:,:,0] = 1 - np.array(aggregate_predictions).T # Transpose as assumed by MI calculating function
            aggregate_results[:,:,1] = np.array(aggregate_predictions).T
            
            # Evaluate uncertainty metrics for aggregate results per participant
            G_X, U_X, log_prob = active_BALD(np.log(aggregate_results), 2)
            
    return results, aggregate_results, aggregate_labels, G_X, U_X


def active_BALD(out, n_classes):
    ''' Calculate prediction entropy and mutual information for a set of MC dropout samples drawn
        at test time. Explanation of metrics can be found on https://arxiv.org/abs/1112.5745.
        Inputs: `out`, output of neural network (y_pred), shape [n_samples, n_feature_windows, n_classes]
                       as a probability
                `n_classes`: number of classes
        Outputs: `G_X`: Predictive entropy
                 `U_X`: Mutual information
                 `log_prob`: log probability of output
    '''
    
    log_prob = np.zeros((out.shape[0], out.shape[1], n_classes))
    score_All = np.zeros((out.shape[1], n_classes))
    All_Entropy = np.zeros((out.shape[1],))
    for d in range(out.shape[0]):
        print ('Dropout Iteration', d)
        log_prob[d] = out[d]
        soft_score = np.exp(log_prob[d])
        score_All = score_All + soft_score
        #computing F_X
        soft_score_log = np.log2(soft_score+10e-15)
        Entropy_Compute = - np.multiply(soft_score, soft_score_log)
        Entropy_Per_samp = np.sum(Entropy_Compute, axis=1)
        All_Entropy = All_Entropy + Entropy_Per_samp
 
    Avg_Pi = np.divide(score_All, out.shape[0])
    Log_Avg_Pi = np.log2(Avg_Pi+10e-15)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
    G_X = Entropy_Average_Pi
    Average_Entropy = np.divide(All_Entropy, out.shape[0])
    F_X = Average_Entropy
    U_X = G_X - F_X
# G_X = predictive entropy
# U_X = MI
    return G_X, U_X, log_prob




def evaluate_uncertainty(G_X, U_X, all_y, all_y_pred, plot=False):
    ''' Evaluate uncertainty thresholds and accuracy on thresholded data according to supplied predictions and
        uncertainty estimates:
        Inputs: `G_X`: predictive entropy
                `U_X`: Mutual information
                `all_y`: True labels, shape [n_x_samples]
                `all_y_pred`: predictions, shape [n_dropout_samples, n_x_samples, n_classes]
                `plot`: (optional), Produce plots to visualise these metrics.
                
        Outputs: `G_X_thresholds_non_empty`: Range of thresholds to plot data over with at least 1 sample present
                 `fraction_remaining_G_X`: Fraction of data remaining at threshold.
                 `test_accs_G_X`: test accuracy according to scikit.learn accuracy_score on remaining data
                 `U_X_thresholds_non_empty`: Range of thresholds to plot data over with at least 1 sample present
                 `fraction_remaining_U_X`: Fraction of data remaining at threshold.
                 `test_accs_U_X`: test accuracy according to scikit.learn accuracy_score on remaining data
    '''

    G_X_thresholds = np.logspace(np.log2(max(G_X + 10**-5)), -20, 500, base=2)
    G_X_thresholds_non_empty = []
    U_X_thresholds = np.logspace(np.log2(max(U_X + 10**-5)), -20, 500, base=2)
    U_X_thresholds_non_empty = []
    

    test_accs_G_X = []
    test_accs_U_X = []
    fraction_remaining_G_X = []
    fraction_remaining_U_X = []
    
    all_y_pred = np.mean(all_y_pred, axis=0)[:,1]
    all_y = np.array(all_y)

    for G_X_threshold in G_X_thresholds:
        G_X_idx =  np.where(G_X<G_X_threshold)[0] # Select indices to threshold
        if G_X_idx.size != 0:  
            if sum(all_y[G_X_idx]) != 0:  # Avoid undefined behaviour where no samples remain with +ve class
                G_X_thresholds_non_empty.append(G_X_threshold)
                test_acc = accuracy_score(all_y[G_X_idx], all_y_pred[G_X_idx] > 0.5) 
                test_accs_G_X.append(test_acc)
                fraction_remaining_G_X.append(len(G_X_idx)/len(all_y))

    for U_X_threshold in U_X_thresholds:
        U_X_idx = np.where(U_X<U_X_threshold)[0]
        if U_X_idx.size != 0:
            if sum(all_y[U_X_idx]) != 0:  # Avoid undefined behaviour where no samples remain with +ve class
                U_X_thresholds_non_empty.append(U_X_threshold) 
                test_acc = accuracy_score(all_y[U_X_idx], all_y_pred[U_X_idx] > 0.5)
                test_accs_U_X.append(test_acc)
                fraction_remaining_U_X.append(len(U_X_idx)/len(all_y))
        

    if plot: # Create two plots if plot = True
        fig, (ax1, ax2) =plt.subplots(2, figsize=(7,7))
        ax1.plot(G_X_thresholds_non_empty, test_accs_G_X)
        ax1.invert_xaxis()
        ax1.set_ylabel('Accuracy on decision')
        ax1.grid()

        ax2.plot(G_X_thresholds_non_empty, fraction_remaining_G_X)
        ax2.invert_xaxis()
        ax2.set_xlabel('Predictive entropy threshold')
        ax2.set_ylabel('Fraction of data remaining')
        ax2.grid()

        plt.show()

        fig, (ax1, ax2) =plt.subplots(2, figsize=(7,7))
        ax1.plot(U_X_thresholds_non_empty, test_accs_U_X)
        ax1.invert_xaxis()
        ax1.set_ylabel('Accuracy on decision')
        ax1.grid()

        ax2.plot(U_X_thresholds_non_empty, fraction_remaining_U_X)
        ax2.invert_xaxis()
        ax2.set_xlabel('Mutual information threshold')
        ax2.set_ylabel('Fraction of data remaining')
        ax2.grid()

        plt.show()

    return G_X_thresholds_non_empty, fraction_remaining_G_X, test_accs_G_X, U_X_thresholds_non_empty, fraction_remaining_U_X, test_accs_U_X

        
def plot_uncertainty(uncertainty, split_type, plot_name=None):
    ''' Plot the uncertainty metrics to reproduce the paper (MI and PE) plots. Requires a dict input which contains the 
        predictions, true labels, and calculated uncertainty metrics (using evaluate_model and active_BALD). This function
        calls evaluate_uncertainty to get the x and y variables for plotting.
        
        Input: `uncertainty`: uncertainty nested dict, with outermost keys defined by split_type, next, audio_modality,
                              and innermost, contains y_pred, y_true, G_X, and U_X.
               `split_type`: Which split type to plot for. Must match the keys in the uncertainty dict. By default, this corresponds to
                             `dict_key` in the main execution code, which defines the train-test pairs.
               `plot_name`: If provided, this will save two figures per function call to
                             (os.path.pardir, 'outputs', plot_name + '_{MI, PE}.pdf')
    '''

    fig, (ax1, ax2) =plt.subplots(2, figsize=(7,7), sharex=True)
    fig2, (ax3, ax4) =plt.subplots(2, figsize=(7,7), sharex=True)


    for audio_modality in uncertainty[split_type].keys():
        y_pred = uncertainty[split_type][audio_modality]['y_pred']
        y_true = uncertainty[split_type][audio_modality]['y_true']
        G_X = uncertainty[split_type][audio_modality]['G_X']
        U_X = uncertainty[split_type][audio_modality]['U_X']

        G_X_thresholds_non_empty, fraction_remaining_G_X, test_accs_G_X, U_X_thresholds_non_empty, fraction_remaining_U_X, test_accs_U_X = evaluate_uncertainty(G_X, U_X, y_true, y_pred, plot=False)



        ax1.plot(G_X_thresholds_non_empty, test_accs_G_X, label=audio_modality)
        ax2.plot(G_X_thresholds_non_empty, fraction_remaining_G_X,  label=audio_modality)
        ax3.plot(U_X_thresholds_non_empty, test_accs_U_X,  label=audio_modality)
        ax4.plot(U_X_thresholds_non_empty, fraction_remaining_U_X,  label=audio_modality)


    ax1.set_title(split_type)

    ax1.invert_xaxis()
    ax1.set_ylabel('Accuracy on decision')
    ax1.grid()

    ax3.invert_xaxis()
    ax3.legend()
    ax3.set_ylabel('Accuracy on decision')
    ax3.grid()

    ax1.legend()
    ax2.set_xlabel('Predictive entropy threshold')
    ax2.set_ylabel('Fraction of data remaining')
    ax2.grid()
    if plot_name:
        fig.savefig(os.path.join(config.results_dir, plot_name + '_PE.pdf'), bbox_inches='tight')
    ax4.set_xlabel('Mutual information threshold')
    ax4.set_ylabel('Fraction of data remaining')
    ax4.grid()
    
    if plot_name:
        fig2.savefig(os.path.join(config.results_dir, plot_name + '_MI.pdf'), bbox_inches='tight')
    plt.show()
    plt.close()





if __name__ == "__main__":
    print('Executing evaluation')
    
    audio_modality_urls =  ['sentence_url', 'exhalation_url_url','cough_url', 'three_cough_url']

    results = defaultdict(dict)
    results_uncertainty = defaultdict(dict)
    # All options for train/test:
    # Trained: naive, test: naive
    # Trained: standard, test: standard, matched, standard long, matched long
    # Trained: matched, test: standard, matched, standard long, matched long

    train_model_test_feat_pairs = [('naive_splits', 'naive_splits_test'), 
                        ('matched', 'matched_test'), ('matched', 'standard_test'), ('matched', 'standard_long'),
                        ('matched', 'matched_long'), ('standard', 'standard_test'), ('standard', 'matched_test'),
                        ('standard', 'standard_long'), ('standard', 'matched_long')]

    for train_type, test_type in train_model_test_feat_pairs:
        dict_key = 'train_' + train_type + '_' + test_type 
        for audio_modality in audio_modality_urls:
            results[dict_key][audio_modality], aggregate_results, aggregate_labels, G_X, U_X = evaluate_model(audio_modality,
                   train_type, test_type, load_part_only=False, n_samples=config.n_samples)
            results_uncertainty[dict_key][audio_modality] = {"y_pred": aggregate_results,
                                                             "y_true": aggregate_labels, "G_X": G_X, "U_X": U_X}

    pd.DataFrame(results).to_csv(os.path.join(config.results_dir, 'main_results.csv'))
    np.save(os.path.join(config.results_dir, 'results_uncertainty.npy'), results_uncertainty)
