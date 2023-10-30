import os
import unittest
import pandas as pd
import numpy as np
from pydub.utils import mediainfo
from tqdm import tqdm

import concurrent.futures

def check_audio_properties(row):
    mismatched_files_local = []
    audio_dir = "/Users/hgc19/uk-covid19-vocal-audio-dataset-open-access-edition/audio"  # Define the audio directory here too, since this function will run in a separate process

    for audio_type in ['exhalation', 'cough', 'three_cough']:
        # Skip if the no audio was collected for that modality
        if row[f"{audio_type}_size"] <= 44.0 or str(row[f"{audio_type}_size"]) == 'nan':
            continue
        file_name_column = f"{audio_type}_file_name"
        file_name = row[file_name_column]
        file_path = os.path.join(audio_dir, str(file_name))
                
        info = mediainfo(file_path)

        try:    
            # Check for the presence of required keys in the info dictionary
            if 'sample_rate' not in info or 'channels' not in info or 'duration' not in info:
                mismatched_files_local.append((file_path, "missing_metadata"))
                continue
            if int(info['sample_rate']) != int(row[f"{audio_type}_sample_rate"]):
                mismatched_files_local.append((file_path, "sample_rate"))
            if int(info['channels']) != int(row[f"{audio_type}_channels"]):
                mismatched_files_local.append((file_path, "channels"))
            if not isclose(float(info['duration']), float(row[f"{audio_type}_length"])):
                logged = row[f"{audio_type}_length"]
                actual = info['duration']
                mismatched_files_local.append((file_path, f"length: logged as: {logged}, actual: {actual}, difference: {abs(float(logged) - float(actual))}"))
        except ValueError as e:
            mismatched_files_local.append((file_path, "ValueError", e))
            continue
        
        
    return mismatched_files_local

def check_audio_properties_chunk(chunk):
    return [check_audio_properties(row) for row in chunk]

def isclose(a, b, abs_tol=0.1):
    # address ploblems with floating point precision
    return abs(a-b) <= abs_tol

class TestAudioFiles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # This will run once before running any test
        cls.audio_metadata = pd.read_csv('/Users/hgc19/uk-covid19-vocal-audio-dataset-open-access-edition/audio_metadata.csv')
        # if row contains missing audio drop the file
        # some recordings are present for some but not all audio recordings
        # cls.audio_metadata = cls.audio_metadata[cls.audio_metadata['missing_audio'] == False]
        cls.audio_dir = "/Users/hgc19/uk-covid19-vocal-audio-dataset-open-access-edition/audio"

    def test_presence_of_files(self):
        missing_files = []
        invalid_files = []
        for index, row in self.audio_metadata.iterrows():
            for audio_type in ['exhalation', 'cough', 'three_cough']:
                audio_type_name = f'{audio_type}_file_name'
                audio_type_size = f'{audio_type}_size'
                file_name = row[audio_type_name]
                
                if row[audio_type_size] <= 44.0 or str(row[audio_type_size]) == 'nan':  # there are cases where participants did not record audio for select modalities. Do not check for their presence
                    continue 
                if not isinstance(file_name, str):  # Check if the filename is a valid string
                    invalid_files.append(f"{file_name}, {row['participant_identifier']}")
                    continue


                file_path = os.path.join(self.audio_dir, file_name)
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
                    
        error_messages = []
        if missing_files:
            error_messages.append(f"The following files are missing: {', '.join(missing_files)}, a total of {len(missing_files)} files are missing")
        if invalid_files:
            error_messages.append(f"The following filenames are invalid: {', '.join(map(str, invalid_files))}, a total of {len(invalid_files)} files are invalid")
        
        self.assertFalse(error_messages, "\n".join(error_messages))

    def test_audio_quality(self):
        # Determine the number of chunks based on the number of available CPU cores
        num_cores = os.cpu_count()
        print(f"Using {num_cores} cores")
        chunks = np.array_split(self.audio_metadata.to_dict('records'), num_cores)
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Wrap the chunks in tqdm to create a progress bar
            results = list(tqdm(executor.map(check_audio_properties_chunk, chunks), total=len(chunks), desc="Processing audio chunks", unit="chunk"))

        # Flatten the results
        mismatched_files = [item for sublist in results for inner_list in sublist for item in inner_list]
        
        self.assertFalse(mismatched_files, f"The following files have mismatched properties: {', '.join([f'{f[0]} ({f[1]})' for f in mismatched_files])}")

if __name__ == '__main__':
    unittest.main()
