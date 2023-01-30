
'''
file to extract opensmile features for laughing and coughing from esc50
author: Harry Coppock
'''
from opensmile_feat_extraction import ExtractOpensmile

import random 
import os
from tqdm import tqdm
import pandas as pd

class MNISTExtract(ExtractOpensmile):
    def __init__(self, path):
        # the constructor is of no use to us
        #super(MNISTExtract, self).__init__():
        self.output_base = './features/opensmile_esc50_binary/'
        self.feature_set = 'compare16/ComParE_2016.conf'
        self.data_path = path
        if not os.path.exists(self.output_base):
            os.makedirs(self.output_base)
        self.files = {}
        self.meta_data = pd.read_csv(os.path.join(self.data_path, 'meta', 'esc50.csv'))
        self.meta_data['labels'] = self.meta_data['category'].apply(lambda x: self.create_labels(x))
        self.create_splits()
        self.iterate_through_files(self.train, 'train')
        self.iterate_through_files(self.test, 'test')

    def create_labels(self, x):
        if x == 'coughing':
            return 'coughing'
        #elif x in ['sneezing', 'breathing', 'laughing', 'snoring']:
        #    return 'other_human'
        else:
            return 'other'

    def create_splits(self):
        '''
        There are 60 speakers therefore create a random split of speaker disjoint train and test sets
        '''
        #self.data = self.meta_data[self.meta_data['labels'].apply(lambda x: x in ['coughing', 'other_human'])]
        self.train = self.meta_data[self.meta_data['fold'] < 4].filename.tolist()
        self.test = self.meta_data[self.meta_data['fold'] >= 4].filename.tolist()

        assert not any([x in self.test for x in self.train]), 'there appears to be cross over between train and test'
        print('train: ', self.train)
        print('test: ', self.test)

    def iterate_through_files(self, dataset, split='train'):
        outfile_base = os.path.join(self.output_base, split)
        outfile = f'{outfile_base}.csv'
        outfile_lld = f'{outfile_base}_lld.csv'
        for file in dataset:
            label = self.meta_data[self.meta_data['filename'] == file].labels.iloc(0)[0]
            print(label)
            self.extract_opensmile(
                    os.path.join(self.data_path, 'audio', file),
                    outfile,
                    label,
                    file)
        
        

if __name__ == '__main__':


    e = MNISTExtract('/workspace/SvmBaseline/ESC-50-master/')
