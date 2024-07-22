import os
from pathlib import Path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(current_dir)
print('Current directory: ', current_dir)

import pandas as pd
import numpy as np
from tqdm import tqdm



# Path to the directory containing the sensor files
data_dir = os.path.join(current_dir, 'data', 'raw')

processed_dir = os.path.join(current_dir, 'data', 'processed', 'etl')

# List of sensor file names
sensores_list = ['PS1','PS2','PS3','PS4','PS5','PS6',
                 'EPS1','FS1','FS2',
                 'TS1','TS2','TS3','TS4',
                 'VS1','CE','CP','SE','profile']

# Dictionary to store the data
X = {}

# Load each sensor file into the dictionary with a progress bar
for s in tqdm(sensores_list, desc="Loading sensor data"):
    file_path = os.path.join(data_dir, s + '.txt')
    X[s] = pd.read_csv(file_path, sep='\t', header=None)

# Just to check the loaded data
for sensor, data in X.items():
    print(f"{sensor} data shape: {data.shape}")
print('\nEnd of importing data\n')

# Each column represents one sample of the data and each row represents one instance, meaning time progresses along the column direction.
# Since each sensor has a different sampling rate, we need to resample the data to have the same sampling rate for all sensors. We will oversample all sensors to 100 Hz.


def oversamp(data, hz):
    """
    Oversamples the given DataFrame to the specified rate.

    Parameters:
    data (pd.DataFrame): The input DataFrame where each column represents a different time series.
    hz (int): The oversampling factor indicating how many times each value should be repeated.
    """
    expanded_data = pd.concat([data] * hz, axis=1)
    expanded_data.columns = [f"{col}_{i+1}" for col in data.columns for i in range(hz)]
    return expanded_data

# Oversamplig
X_dict = {}
for k, v in {'TS1': 100, 'TS2': 100, 'TS3': 100, 'TS4': 100, 'VS1':100, 'CE': 100,'CP': 100, 'SE': 100, 'FS1': 10, 'FS2': 10}.items():
    X_dict[k] = oversamp(X[k], v)
    
# Add the rest of the sensors to the dictionary
for c in ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1', 'profile']:
    X_dict[c] = X[c].copy()
    
print('\nOversampled data:\n')
for key, df in X_dict.items():
    print(f"Key: {key}, Size: {df.shape}")
    
# Function to save data to parquet
def save_to_parquet(data_dict, directory):
    for key, data in data_dict.items():
        file_path = os.path.join(directory, f'{key}.parquet')
        data.columns = [str(col) for col in data.columns]  # Ensure column names are strings
        data.to_parquet(file_path, index=False)
    
        
save_to_parquet(X_dict, processed_dir)
print('\nData saved to parquet format\n')