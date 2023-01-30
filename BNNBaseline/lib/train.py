from datetime import datetime
from collections import defaultdict
import pickle
import os

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

import numpy as np
from sklearn.metrics import accuracy_score

from load_feat import load_data
import config
import models


def build_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size=64, model_method='resnet'):
    ''' Build dataloaders for training neural networks.
        Inputs: `x_train`: Array of features to train. Shape as assumed at output of feature extraction code.
                `y_train`: Labels for feature array
                `x_val`: Array of features for validation. Shape as assumed at output of feature extraction code.
                `x_test`: Array of features for test loop (following training)
                `y_test`: Labels for feature array
                `batch_size`: Batch size for training, configurable in config.py
        Outputs: `train_loader`: DataLoader of TensorDataset for use in PyTorch for training
                 `val_loader`: DataLoader of TensorDataset for use in PyTorch for validation
                 `test_loader`: DataLoader of TensorDataset for use in PyTorch for testing
     '''
    
    
    if model_method == 'resnet':
        # Replicate across 3 channels for ResNet assumption. Memory intensive. May have to generate data on fly
        # or load from parts if RAM limited
        
        x_train = torch.tensor(x_train).float()
        y_train = torch.tensor(y_train).unsqueeze(1).float()       
        x_val = torch.tensor(x_val).float()
        y_val = torch.tensor(y_val).unsqueeze(1).float()
        x_test = torch.tensor(x_test).float()
        y_test = torch.tensor(y_test).unsqueeze(1).float()
        
        x_train = x_train.unsqueeze(1).repeat(1,3,1,1)
        x_val = x_val.unsqueeze(1).repeat(1,3,1,1)
        x_test = x_test.unsqueeze(1).repeat(1,3,1,1)
        print('Train (x, y), val, test, tensor shapes, dtype:', np.shape(x_train), np.shape(y_train),
              np.shape(x_val), np.shape(y_val), np.shape(x_test), np.shape(y_test), x_train.dtype)
        

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        test_dataset = TensorDataset(x_test, y_test)
        
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    elif model_method == 'vgg':
        # Implement different models (e.g. vggish) here and write own dataloader
        print('Loading vgg style feats')
    
    return train_loader, val_loader, test_loader




def test_model(model, test_loader, criterion, class_threshold=0.5, device=None):
    ''' Return accuracy score for ...
        Inputs: `model`: model object, by default ResNet-50-dropout model
                `test_loader`: test loader as supplied by build_dataloaders()
                `criterion`: loss function
                `class_threshold`: class threshold for accuracy score reporting
                `device`: (optional) device for evaluation.
    '''
    
    # Instantiate device
    if device is None:
        torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    test_loss = 0.0
    model.eval()  # Ensure model in .eval() mode. Note that with the current
    # implementation of the BNN-ResNet, custom dropout layers remain enabled
    # which will give stochastic preformance over test samples
    
    all_y = []
    all_y_pred = []
    for inputs in test_loader:
        ##Necessary in order to handle single and multi input feature spaces

        x = inputs[0].to(device).detach()
        y = inputs[1].to(device).detach()

        
        y_pred = model(x)
        
        loss = criterion(y_pred, y)

        test_loss += loss.item()
        
        all_y.append(y.cpu().detach())
        all_y_pred.append(y_pred.cpu().detach())
        
        del x
        del y
        del y_pred
        
    all_y = torch.cat(all_y)
    all_y_pred = torch.cat(all_y_pred)
    
    test_loss = test_loss/len(test_loader)
    test_acc = accuracy_score(all_y.numpy(), (all_y_pred.numpy() > class_threshold).astype(float))
    
    
    return test_loss, test_acc


def traintest(x_train, y_train, x_val, y_val, x_test, y_test, audio_modality, split_type, model_method = 'resnet'):
    ''' Main function to train models according to the specified split_type, and audio_modality. Requires a valid
        validation set. The default model method is the BNN-ResNet-50 dropout model as imported from models. This 
        function is parameterised in the config.py file for learning rate, batch size, and other options.
        The function saves the best model checkpoints according the validation accuracy and the corresponding
        output dict to `os.path.join(os.path.pardir, 'models', split_type, audio_modality)`.
        
        Input: `x_train`: Input features required, in compatibility with output from previous pipeline stage
               `y_train`: Corresponding labels of input features.
               `x_val`: Validation feature set
               `y_val`: Labels for validation data
               `x_test`: Test set for the split. Test data will be automatically evaluated over with the best
                         model according to the validation accuracy criterion during training. No optimisation
                         over the test sets is performed.
               `y_test`: Labels for corresponding test set.
               `audio_modality`: Audio modality for loading pickled features. By default in:
                                 ['sentence_url', 'exhalation_url_url', 'cough_url', 'three_cough_url']
               `split_type`: Split type required for loading pickled features. By default in:
                                 ['naive_splits', 'matched', 'standard']
               `model_method`: This currently supports only the ResNet, but you can add your own model classes here
         Outputs: `outputs`: A dict of training and test metrics, returned at the end of the full training loop.  
                             Importantly, the models and outputs are saved automatically to disk in 
                             `os.path.join(os.path.pardir, 'models', split_type, audio_modality)`
                                outputs['all_train_loss'] = all_train_loss
                                outputs['all_train_acc'] = all_train_acc
                                outputs['all_val_loss'] = all_val_loss
                                outputs['all_val_acc'] = all_val_acc
                                outputs['test_loss'] = test_loss
                                outputs['test_acc'] = test_acc
                                outputs['best_epoch'] = best_epoch
    '''
             
    batch_size = config.batch_size
    epochs = config.epochs
    max_overrun = config.max_overrun
    class_threshold = config.class_threshold
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')
    
    # Initialise dict for outputs
    outputs = defaultdict(dict)
    
    # Build train, val and test dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size=batch_size)

    # Define model here
    model = models.ResnetDropoutFull()
    model = model.to(device)

    criterion = nn.BCELoss()
    optimiser = optim.Adam(model.parameters(), lr=config.lr)
    
    
    # Initialise all results variables
    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []
    best_val_loss = np.inf
    best_val_acc = -np.inf
    best_epoch = -1
    checkpoint_name = None
    overrun_counter = 0
    
    # Begin batch training over epochs
    for e in range(epochs):
        train_loss = 0.0
        model.train()

        all_y = []
        all_y_pred = []
        for batch_i, inputs in enumerate(train_loader):

            x = inputs[0].to(device).detach()
            y = inputs[-1].to(device).detach()

            optimiser.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)

            loss.backward()
            optimiser.step()

            train_loss += loss.item()
            all_y.append(y.cpu().detach())
            all_y_pred.append(y_pred.cpu().detach())

            del x
            del y

        all_train_loss.append(train_loss/len(train_loader))

        all_y = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred)
        train_acc = accuracy_score(all_y.numpy(), (all_y_pred.numpy() > class_threshold).astype(float))
        all_train_acc.append(train_acc)

        val_loss, val_acc = test_model(model, val_loader, criterion, class_threshold, device=device)
        all_val_loss.append(val_loss)
        all_val_acc.append(val_acc)

        if val_acc > best_val_acc: # Define model selection as model that performs best on custom validation data

            if checkpoint_name is not None:
                os.path.join(os.path.pardir, 'models', split_type, audio_modality, checkpoint_name)

            checkpoint_name = f'model_{split_type}_{audio_modality}_e{e}_{model_method}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pth'
            if not os.path.exists(os.path.join(os.path.pardir, 'models', split_type, audio_modality)):
                os.makedirs(os.path.join(os.path.pardir, 'models', split_type, audio_modality))
            ##TODO: not strictly necessary, can just hold the model in memory until early stopping finishes.
            torch.save(model.state_dict(), os.path.join(os.path.pardir, 'models', split_type, audio_modality, checkpoint_name))

            best_epoch = e
            best_val_acc = val_acc
            overrun_counter = -1

        overrun_counter += 1

        print('Epoch: %d, Train Loss: %.8f, Train Acc: %.8f Val Loss: %.8f, Val Acc: %.4f, %d' % (e, train_loss/len(train_loader), train_acc, val_loss, val_acc, overrun_counter))

        if overrun_counter > max_overrun:  # Stop if number of epochs exceeded with no improvement in chosen metric
            break
                
    ## Evaluate best model
    model.load_state_dict(torch.load(os.path.join(os.path.pardir, 'models', split_type, audio_modality, checkpoint_name)))
    test_loss, test_acc = test_model(model, test_loader, criterion, class_threshold, device=device)

    print('Best Epoch: %d, Test Loss: %.8f, Test Acc: %.4f' % (best_epoch, test_loss, test_acc))

    outputs['all_train_loss'] = all_train_loss
    outputs['all_train_acc'] = all_train_acc
    outputs['all_val_loss'] = all_val_loss
    outputs['all_val_acc'] = all_val_acc
    outputs['test_loss'] = test_loss
    outputs['test_acc'] = test_acc
    outputs['best_epoch'] = best_epoch
        
        

    output_name = f'outputs_{split_type}_{audio_modality}_{model_method}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
    with open(os.path.join(os.path.pardir, 'models', split_type, audio_modality, output_name), 'wb') as fp:
        pickle.dump(outputs, fp)  # Save results for model object in same folder as model output
        
    return outputs


if __name__ == '__main__':
    # Experiment lists to do for training:
    # train: standard, train. val: standard, val. test: standard, test; match, test, standard, long; match, long
    # train: match, train. val: match, val, test: standard, test; match, test, standard, long; match, long
    # train: naive, train. val: naive, val, test: naive
    outputs = {}
    data_type = ['train', 'val', 'test', 'long']
    for split_type in ['naive_splits', 'matched', 'standard']:
        for audio_modality in ['sentence_url', 'exhalation_url_url', 'cough_url', 'three_cough_url']:

            x_train, y_train = load_data(audio_modality, split_type + '_' + data_type[0], load_part_only = False)
            x_val, y_val = load_data(audio_modality, split_type + '_' + data_type[1], load_part_only = False)
            # Test partition not important as will evaluate separately
            x_test, y_test = load_data(audio_modality, split_type + '_' + data_type[2], load_part_only = False)

            # Train in loop and select model with highest validation 
            outputs[split_type + '_' + audio_modality] = traintest(x_train, y_train, x_val, y_val,
                                                            x_test, y_test, audio_modality, split_type) 