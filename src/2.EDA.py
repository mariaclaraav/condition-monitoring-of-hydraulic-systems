import os
from pathlib import Path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(current_dir)
print('Current directory: ', current_dir)
print()

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from src.utils.statistics import fill_outliers, describe_data
from src.utils.data_manipulation import save_to_parquet

plt.rcParams['font.family'] = 'Times New Roman'


data_path = os.path.join(current_dir, 'data', 'processed', 'etl')
# List of sensor file names
sensores_list = ['PS1','PS2','PS3','PS4','PS5','PS6',
                 'EPS1','FS1','FS2',
                 'TS1','TS2','TS3','TS4',
                 'VS1','CE','CP','SE','profile']

# Dictionary to store the data
X_dict = {}

# Load sensor data from parquet files
for s in tqdm(sensores_list, desc="Loading sensor data"):
    file_path = os.path.join(data_path, s + '.parquet')
    X_dict[s] = pd.read_parquet(file_path)

# Just to check the loaded data
for sensor, data in X_dict.items():
    print(f"{sensor} data shape: {data.shape}")

# separete the profile data
X_dict['profile'].rename(columns={'2': 'profile'}, inplace=True)
X_profile = X_dict['profile']['profile']
X_profile = pd.DataFrame(X_profile)

if 'profile' in X_dict:
    del X_dict['profile']
print('\nChecking and removing outliers...\n')
# Remove outliers and replace with the mean
for k, v in X_dict.items():
    X_dict[k] = fill_outliers(v, k, c = 2.5)

print('\nCreating descriptive and statistics features...\n')
# Get the descriptive statistics
for k, v in X_dict.items():
    X_dict[k] = describe_data(v)

print('\nSaving new data...\n')
processed_dir = os.path.join(current_dir, 'data', 'processed', 'features')
save_to_parquet(X_dict, processed_dir)

# Save X_profile separately
profile_file_path = os.path.join(processed_dir, 'profile.parquet')
X_profile.to_parquet(profile_file_path, index=False)
X_profile
print('\nData saved to parquet format\n')