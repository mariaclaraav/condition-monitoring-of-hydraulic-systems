import os

def save_to_parquet(data_dict, directory):
    for key, data in data_dict.items():
        file_path = os.path.join(directory, f'{key}.parquet')
        data.columns = [str(col) for col in data.columns]  # Ensure column names are strings
        data.to_parquet(file_path, index=False)
    