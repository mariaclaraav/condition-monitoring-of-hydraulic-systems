import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Define the current directory and add it to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(current_dir)

print('Current directory: ', current_dir)
print()

from src.utils.models import ModelExperiment

# Define the path to the processed data
data_path = os.path.join(current_dir, 'data', 'processed', 'features')
model_path = os.path.join(current_dir, 'src', 'models')
# List of sensor file names
sensores_list = [
    'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6',
    'EPS1', 'FS1', 'FS2',
    'TS1', 'TS2', 'TS3', 'TS4',
    'VS1', 'CE', 'CP', 'SE', 'profile'
]

# Dictionary to store the data
X_dict = {}

# Load sensor data from parquet files
for s in tqdm(sensores_list, desc="Loading sensor data"):
    file_path = os.path.join(data_path, s + '.parquet')
    X_dict[s] = pd.read_parquet(file_path)

# List of selected keys to rename columns
selected_keys = ['TS1', 'TS2', 'VS1', 'SE', 'FS1', 'EPS1', 'PS1', 'PS2', 'PS3']

# Function to rename columns
def rename_columns(df, key):
    new_columns = {col: f"{key}_{col}" for col in df.columns}
    df.rename(columns=new_columns, inplace=True)
    return df

# Rename columns for selected keys
for key in selected_keys:
    if key in X_dict:
        X_dict[key] = rename_columns(X_dict[key], key)

# Concatenate the final DataFrame
X_final = pd.concat(
    [X_dict['TS1'], X_dict['TS2'], X_dict['VS1'],
     X_dict['SE'], X_dict['FS1'], X_dict['EPS1'],
     X_dict['PS1'], X_dict['PS2'], X_dict['PS3'], X_dict['profile']],
    axis=1
)



X_train, X_val_test, y_train, y_val_test = train_test_split(X_final.iloc[:,:-1], X_final.iloc[:,-1], train_size=0.6, random_state=42, shuffle=False)

X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42, shuffle=False)


experiment = ModelExperiment(X_train, y_train, X_val, y_val, only_execute=['Logistic Regression', 'Naive Bayes'], search_execute=['Decision Tree', 'Random Forest', 'Neural Net', 'AdaBoost'])
results = experiment.run_experiments(save_dir=model_path)

