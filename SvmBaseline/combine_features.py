import pandas as pd
from glob import glob
import os
from tqdm import tqdm


POSSIBLE_MODALITIES = ['audio_ha_sound_url',
                        'audio_cough_url',
                        'audio_three_cough_url',
                        'audio_sentence_url']
def combine_features():
    '''
    combines the extracted opensmile features
    '''
    
    for i, modality in tqdm(enumerate(POSSIBLE_MODALITIES)):
        if i == 0:
            train_combined, test_combined = load_dataset_modality(modality)
            # there appear to be duplicates in the test set - this is a temp hack 
            # until the issue is sorted
            test_combined.drop_duplicates(inplace=True, subset=[0])
        else:
            train_temp, test_temp = load_dataset_modality(modality)
            test_temp.drop_duplicates(inplace=True, subset=[0])
            train_combined = pd.merge(train_combined, train_temp, on=[0, 6374])
            test_combined = pd.merge(test_combined, test_temp, on=[0, 6374])
        print('train:', train_combined.shape, '. Number unique ids=', len(train_combined[0].unique()))
        print('train', train_combined.columns)
        print('train', train_combined)
        print('test:', test_combined.shape, '. Number unique ids=', len(test_combined[0].unique()))
        print('test', test_combined.columns)
        print('test', test_combined)
    train_combined.rename({6374:'label'})
    test_combined.rename({6374:'label'})
    
    train_combined.to_csv('features/opensmile/combined_train.csv')
    test_combined.to_csv('features/opensmile/combined_test.csv')
    ##train_X = train_df.values[:, 1:label_index].astype(np.float32)
    #train_y = train_df.values[:, label_index].astype(str)

def load_dataset_modality(modality):
    '''
    return the train and test sets for the given modality
    '''
    train_file = glob(os.path.join('features/opensmile', modality, 'train.*'))[0]
    test_file = glob(os.path.join('features/opensmile', modality, 'test.*'))[0]
    train_df = pd.read_csv(train_file, skiprows=6379, header=None)
    test_df = pd.read_csv(test_file, skiprows=6379, header=None)
    return train_df, test_df

if __name__ =='__main__':
   combine_features() 
