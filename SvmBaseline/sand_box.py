import os
import pandas as pd



for filename in os.listdir('features/opensmile/train/'):
    data = pd.read_csv(f'features/opensmile/train/{filename}')
    print(data.head())
