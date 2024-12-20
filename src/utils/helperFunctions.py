import pandas as pd
from pathlib import Path
import os
import pickle
import dill

def create_kaggle_results(predictions, path_to_kaggledata='./data/test_data-Kaggle-v0.11.csv', csv_name:str='kaggle_results.csv'):
    '''
    Creates a csv file with the predictions for the Kaggle competition.
    '''
    indexes = pd.read_csv(path_to_kaggledata)['Unnamed: 0']
    results = pd.DataFrame()
    results['Id'] = indexes
    results['Expected'] = predictions

    os.makedirs('./results/kaggle', exist_ok=True)

    date_now = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
    csv_name = f'{csv_name}_{date_now}.csv'

    results.to_csv(f'./results/kaggle/{csv_name}', index=False)
    print(f'File {csv_name} created successfully.')


def pickle_dataframe(df: pd.DataFrame, file_name: str):
    """
    Save a pandas DataFrame as a pickle or dill file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - file_name (str): The name of the file (with extension).
    - use_dill (bool): Whether to use dill instead of pickle.
    """
    with open(file_name, 'wb') as file:
        dill.dump(df, file)
    print(f"DataFrame saved to {file_name}.")


def unpickle_dataframe(file_name: str) -> pd.DataFrame:
    """
    Load a pandas DataFrame from a pickle or dill file.

    Parameters:
    - file_name (str): The name of the file to load.
    - use_dill (bool): Whether to use dill instead of pickle.

    Returns:
    - pd.DataFrame: The loaded DataFrame.
    """
    with open(file_name, 'rb') as file:
        df = dill.load(file)
    print(f"DataFrame loaded from {file_name}.")
    return df