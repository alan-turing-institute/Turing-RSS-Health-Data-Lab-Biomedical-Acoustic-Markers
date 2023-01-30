'''
file to extract opensmile features for the spoken mnist dataset
author: Harry Coppock
'''
from opensmile_feat_extraction import ExtractOpensmile

import random 
import os
from tqdm import tqdm

class MNISTExtract(ExtractOpensmile):
    def __init__(self, path):
        # the constructor is of no use to us
        #super(MNISTExtract, self).__init__():
        self.output_base = './features/opensmile_mnist/'
        self.feature_set = 'compare16/ComParE_2016.conf'
        self.data_path = path
        if not os.path.exists(self.output_base):
            os.makedirs(self.output_base)
        self.files = {}
        self.create_splits()
        self.iterate_through_files(self.train, 'train')
        self.iterate_through_files(self.test, 'test')
        assert not any([x in self.files['test'] for x in self.files['train']]), 'there appears to be file cross over between train and test'

    def create_splits(self):
        '''
        There are 60 speakers therefore create a random split of speaker disjoint train and test sets
        '''
        users = os.listdir(self.data_path)
        users = [user for user in users if os.path.isdir(os.path.join(self.data_path, user)) and 'checkpoint' not in user]
        print(users)
        print(f'There are {len(users)} users')
        self.train = ['28', '56', '07', '19', '35', '01', '06', '16', '23', '34', '46', '53', '36', '57', '09', '24', '37', '02', \
                    '08', '17', '29', '39', '48', '54', '43', '58', '14', '25', '38', '03', '10', '20', '30', '40', '49', '55', \
                    '12', '47', '59', '15', '27', '41', '04', '11', '21', '31', '44', '50']
        self.test = ['26', '52', '60', '18', '32', '42', '05', '13', '22', '33', '45', '51']
        assert not any([x in self.test for x in self.train]), 'there appears to be cross over between train and test'
        print('train: ', self.train)
        print('test: ', self.test)

    def iterate_through_files(self, dataset, split='train'):
        outfile_base = os.path.join(self.output_base, split)
        outfile = f'{outfile_base}.csv'
        outfile_lld = f'{outfile_base}_lld.csv'
        self.files[split] = []
        
        for user in tqdm(dataset):
            for file in os.listdir(
                    os.path.join(self.data_path, user)):
                if not file.endswith('.wav'):
                    continue
                self.files[split].append(file)
                label = file[:1] #file digit of file specifies the spoken word
                self.extract_opensmile(
                        os.path.join(self.data_path, user, file),
                        outfile,
                        label,
                        file)

if __name__ == '__main__':


    e = MNISTExtract('/workspace/SvmBaseline/AudioMNIST/data')





