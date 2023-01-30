'''
author: Harry Coppock
Qs to: harry.coppock@imperial.ac.uk
'''
import unittest
import pandas as pd
import sys
sys.path.append('../')
sys.path.append('../../')
from utils.dataset_stats import DatasetStats

class TestSplitsSingleFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extract = DatasetStats()
        cls.meta = cls.extract.meta_data
    
    def test_no_duplicates_train(self):
        print('Test 1')
        self.assertEqual(
                len(self.meta[self.meta['splits'] == 'train'].participant_identifier.unique()), 
                len(self.meta[self.meta['splits'] == 'train'].participant_identifier),
                f'''There appear to be duplicate barcodes in the train set. Unique: {len(self.meta[self.meta['splits'] == 'train'].participant_identifier.unique())}, actual: {len(self.meta[self.meta['splits'] == 'train'].participant_identifier)}'''
                        ) 
    def test_no_duplicates_val(self):
        print('Test 2')
        self.assertEqual(
                len(self.meta[self.meta['splits'] == 'val'].participant_identifier.unique()), 
                len(self.meta[self.meta['splits'] == 'val'].participant_identifier),
                f'''There appear to be duplicate barcodes in the val set. Unique: {len(self.meta[self.meta['splits'] == 'val'].participant_identifier.unique())}, actual: {len(self.meta[self.meta['splits'] == 'val'].participant_identifier)}'''
                        ) 
    def test_no_duplicates_test(self):
        print('Test 3')
        self.assertEqual(
                len(self.meta[self.meta['splits'] == 'test'].participant_identifier.unique()), 
                len(self.meta[self.meta['splits'] == 'test'].participant_identifier),
                f'''There appear to be duplicate barcodes in the test set. Unique: {len(self.meta[self.meta['splits'] == 'test'].participant_identifier.unique())}, actual: {len(self.meta[self.meta['splits'] == 'test'].participant_identifier)}'''
                        ) 
    def test_no_duplicates_long(self):
        print('Test 4')
        self.assertEqual(
                len(self.meta[self.meta['splits'] == 'long'].participant_identifier.unique()), 
                len(self.meta[self.meta['splits'] == 'long'].participant_identifier),
                f'''There appear to be duplicate barcodes in the long set. Unique: {len(self.meta[self.meta['splits'] == 'long'].participant_identifier.unique())}, actual: {len(self.meta[self.meta['splits'] == 'long'].participant_identifier)}'''
                        ) 
    def test_crossover_train_tests(self):
        print('Test 5')
        self.assertFalse(
                any(check in self.meta[self.meta['splits'] == 'test'].participant_identifier.tolist() for check in self.meta[self.meta['splits'] == 'train'].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both train and test,
                             investigate!''') 
    def test_crossover_train_val(self):
        print('Test 6')
        self.assertFalse(
                any(check in self.meta[self.meta['splits'] == 'val'].participant_identifier.tolist() for check in self.meta[self.meta['splits'] == 'train'].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both train and val,
                             investigate!''') 
    def test_crossover_train_long(self):
        print('Test 7')
        self.assertFalse(
                any(check in self.meta[self.meta['splits'] == 'long'].participant_identifier.tolist() for check in self.meta[self.meta['splits'] == 'train'].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both train and long,
                             investigate!''') 
    def test_crossover_val_tests(self):
        print('Test 8')
        self.assertFalse(
                any(check in self.meta[self.meta['splits'] == 'test'].participant_identifier.tolist() for check in self.meta[self.meta['splits'] == 'val'].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both val and test,
                             investigate!''') 
    def test_crossover_val_long(self):
        print('Test 9')
        self.assertFalse(
                any(check in self.meta[self.meta['splits'] == 'long'].participant_identifier.tolist() for check in self.meta[self.meta['splits'] == 'val'].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both val and long,
                             investigate!''')
    def test_crossover_long_tests(self):
        print('Test 10')
        self.assertFalse(
                any(check in self.meta[self.meta['splits'] == 'test'].participant_identifier.tolist() for check in self.meta[self.meta['splits'] == 'long'].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both long and test,
                             investigate!''')

    
    def test_no_duplicates_train_orig(self):
        print('Test 11')
        self.assertEqual(
                len(self.meta[self.meta['original_splits'] == 'train'].participant_identifier.unique()), 
                len(self.meta[self.meta['original_splits'] == 'train'].participant_identifier),
                f'''There appear to be duplicate barcodes in the train set. Unique: {len(self.meta[self.meta['original_splits'] == 'train'].participant_identifier.unique())}, actual: {len(self.meta[self.meta['original_splits'] == 'train'].participant_identifier)}'''
                        ) 
    def test_no_duplicates_val_orig(self):
        print('Test 12')
        self.assertEqual(
                len(self.meta[self.meta['original_splits'] == 'val'].participant_identifier.unique()), 
                len(self.meta[self.meta['original_splits'] == 'val'].participant_identifier),
                f'''There appear to be duplicate barcodes in the val set. Unique: {len(self.meta[self.meta['original_splits'] == 'val'].participant_identifier.unique())}, actual: {len(self.meta[self.meta['original_splits'] == 'val'].participant_identifier)}'''
                        ) 
    def test_no_duplicates_test_orig(self):
        print('Test 13')
        self.assertEqual(
                len(self.meta[self.meta['original_splits'] == 'test'].participant_identifier.unique()), 
                len(self.meta[self.meta['original_splits'] == 'test'].participant_identifier),
                f'''There appear to be duplicate barcodes in the test set. Unique: {len(self.meta[self.meta['original_splits'] == 'test'].participant_identifier.unique())}, actual: {len(self.meta[self.meta['original_splits'] == 'test'].participant_identifier)}'''
                        ) 
    def test_no_duplicates_long_orig(self):
        print('Test 14')
        self.assertEqual(
                len(self.meta[self.meta['original_splits'] == 'long'].participant_identifier.unique()), 
                len(self.meta[self.meta['original_splits'] == 'long'].participant_identifier),
                f'''There appear to be duplicate barcodes in the long set. Unique: {len(self.meta[self.meta['original_splits'] == 'long'].participant_identifier.unique())}, actual: {len(self.meta[self.meta['original_splits'] == 'long'].participant_identifier)}'''
                        ) 
    def test_crossover_train_tests_orig(self):
        print('Test 15')
        self.assertFalse(
                any(check in self.meta[self.meta['original_splits'] == 'test'].participant_identifier.tolist() for check in self.meta[self.meta['original_splits'] == 'train'].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both train and test,
                             investigate!''') 
    def test_crossover_train_val(self):
        print('Test 16')
        self.assertFalse(
                any(check in self.meta[self.meta['original_splits'] == 'val'].participant_identifier.tolist() for check in self.meta[self.meta['original_splits'] == 'train'].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both train and val,
                             investigate!''') 
    def test_crossover_train_long_orig(self):
        print('Test 17')
        self.assertFalse(
                any(check in self.meta[self.meta['original_splits'] == 'long'].participant_identifier.tolist() for check in self.meta[self.meta['original_splits'] == 'train'].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both train and long,
                             investigate!''') 
    def test_crossover_val_tests_orig(self):
        print('Test 18')
        self.assertFalse(
                any(check in self.meta[self.meta['original_splits'] == 'test'].participant_identifier.tolist() for check in self.meta[self.meta['original_splits'] == 'val'].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both val and test,
                             investigate!''') 
    def test_crossover_val_long_orig(self):
        print('Test 19')
        self.assertFalse(
                any(check in self.meta[self.meta['original_splits'] == 'long'].participant_identifier.tolist() for check in self.meta[self.meta['original_splits'] == 'val'].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both val and long,
                             investigate!''') 
    def test_crossover_long_tests_orig(self):
        print('Test 20')
        self.assertFalse(
                any(check in self.meta[self.meta['original_splits'] == 'test'].participant_identifier.tolist() for check in self.meta[self.meta['original_splits'] == 'long'].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both long and test,
                             investigate!''')

    def test_no_duplicates_matched_train(self):
        print('Test 26')
        self.assertEqual(
                len(self.meta[self.meta['matched_train_splits'] == 'matched_train'].participant_identifier.unique()), 
                len(self.meta[self.meta['matched_train_splits'] == 'matched_train'].participant_identifier),
                f'''There appear to be duplicate barcodes in the matched train set.'''
                        ) 
    def test_no_duplicates_matched_validation(self):
        print('Test 27')
        self.assertEqual(
                len(self.meta[self.meta['matched_train_splits'] == 'matched_validation'].participant_identifier.unique()), 
                len(self.meta[self.meta['matched_train_splits'] == 'matched_validation'].participant_identifier),
                f'''There appear to be duplicate barcodes in the matched validation set.'''
                        )
        
    def test_no_duplicates_matched_long(self):
        print('Test 28')
        self.assertEqual(
                len(self.meta[self.meta['stratum_matched_original_long_test'] == True].participant_identifier.unique()), 
                len(self.meta[self.meta['stratum_matched_original_long_test'] == True].participant_identifier),
                f'''There appear to be duplicate barcodes in the matched train set.'''
                        ) 

    def test_no_duplicates_matched_train_orig(self):
        print('Test 29')
        self.assertEqual(
                len(self.meta[self.meta['matched_original_train_splits'] == 'matched_train'].participant_identifier.unique()), 
                len(self.meta[self.meta['matched_original_train_splits'] == 'matched_train'].participant_identifier),
                f'''There appear to be duplicate barcodes in the matched train set.'''
                        ) 
    def test_no_duplicates_matched_validation_orig(self):
        print('Test 26')
        self.assertEqual(
                len(self.meta[self.meta['matched_original_train_splits'] == 'matched_validation'].participant_identifier.unique()), 
                len(self.meta[self.meta['matched_original_train_splits'] == 'matched_validation'].participant_identifier),
                f'''There appear to be duplicate barcodes in the matched validation set.'''
                        )
    
    def test_crossover_matched_train_tests(self):
        print('Test 27')
        self.assertFalse(
                any(check in self.meta[self.meta['splits'] == 'test'].participant_identifier.tolist() for check in self.meta[self.meta['matched_train_splits'] == 'matched_train'].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both matched train and test,
                             investigate!''') 
    def test_crossover_matched_val_tests(self):
        print('Test 28')
        self.assertFalse(
                any(check in self.meta[self.meta['splits'] == 'test'].participant_identifier.tolist() for check in self.meta[self.meta['matched_train_splits'] == 'matched_validation'].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both matched validation and test,
                             investigate!''') 
    def test_crossover_train_matchedtests(self):
        print('Test 29')
        self.assertFalse(
                any(check in self.meta[self.meta['splits'] == 'train'].participant_identifier.tolist() for check in self.meta[self.meta['in_matched_rebalanced_test'] == True].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both train and matched test,
                             investigate!''') 
    def test_crossover_val_matchedtests(self):
        print('Test 30')
        self.assertFalse(
                any(check in self.meta[self.meta['splits'] == 'val'].participant_identifier.tolist() for check in self.meta[self.meta['in_matched_rebalanced_test'] == True].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both validation and matched test,
                             investigate!''') 
    def test_crossover_val_matched_train_matched(self):
        print('Test 31')
        self.assertFalse(
                any(check in self.meta[self.meta['matched_train_splits'] == 'matched_validation'].participant_identifier.tolist() for check in self.meta[self.meta['matched_train_splits'] == 'matched_train'].participant_identifier.tolist()),
                        f'''There appears to be unique barcode ids in both validation matched and matched validation, investigate!''')


    def test_no_void_results_train(self):
        print('Test 32')
        self.assertFalse(
                'Unknown/Void' in self.meta[self.meta['splits'] == 'train'].covid_test_result.unique().tolist(),
                f'''There appears to be void results in train''')
    def test_no_void_results_test(self):
        print('Test 33')
        self.assertFalse(
                'Unknown/Void' in self.meta[self.meta['splits'] == 'test'].covid_test_result.unique().tolist(),
                f'''There appears to be void results in test''')
    def test_no_void_results_validation(self):
        print('Test 34')
        self.assertFalse(
                'Unknown/Void' in self.meta[self.meta['splits'] == 'validation'].covid_test_result.unique().tolist(),
                f'''There appears to be void results in validation''')
    def test_no_void_results_long(self):
        print('Test 35')
        self.assertFalse(
                'Unknown/Void' in self.meta[self.meta['splits'] == 'long'].covid_test_result.unique().tolist(),
                f'''There appears to be void results in long''')
if __name__ == '__main__':
    unittest.main()
