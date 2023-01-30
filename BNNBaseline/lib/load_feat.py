import numpy as np
import config
import os
    
def load_data(modality_url, feat_name, load_part_only=False, stacked=True):
    ''' Load data from disk in pickle format.
        Inputs: `modality_url`: Audio modality url, as indexed from original metadata dataframe.
                `feat_name`: Feature name to load, corresponding to outputs of feature extraction pipeline.
                `load_part_only`: Only load a single pickle part for debug purposes.
                `stacked`: If True, return a stacked feature window array with the first dimension being 
                           sum(n_features_per_recording). If False, returns an object of nested lists.
        Outputs: `x`: feature array
                 `y`: label array
    '''
    print(os.getcwd())    
    # Example naming convention:
    # Matching function name convention def get_feat(audio_url_list, label_list, bucket_name, modality_url, output_dir=None,
    #        n_samples_per_pickle=3000, feat_name=None):
    #     np.save('../../outputs/feats/' + feat_name + '_feat_list_' + modality + '_' + str(i) + '.npy', feat_list)
    x = []
    y = []
    
    if load_part_only:
        n = 1 # Load only one single pickle part for debug purposes
    else:
        n = 100 # Search for all pickle parts up to n. By default each pickle contains 3000 samples, and there are at most ~
                # 15 parts per array, of one GB each.
    for i in range(n):
        print('Attempting to load', config.feat_dir + feat_name + '_feat_list_' + modality_url + '_' + str(i) + '.npy')

        try:
            # naive_splits_val_feat_list_cough_url_0.npy
            x.append(np.load(config.feat_dir + feat_name + '_feat_list_' + modality_url + '_' + str(i) + '.npy',
                             allow_pickle=True))
            y.append(np.load(config.feat_dir + feat_name + '_label_list_' + modality_url + '_' + str(i) + '.npy',
                             allow_pickle=True))
            print('Loaded index', i)
        except:
            print('Loaded', i, 'files')
            break

    if stacked:
        x = np.vstack(np.hstack(x))
        y = np.hstack(np.hstack(y))
    else:
        x = np.hstack(x)
        y = np.hstack(y)
    
    print('Loaded feat shape', np.shape(x))
    print('Loaded label shape', np.shape(y))
    return x, y
