import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle
import dill
from sklearn.metrics import mean_absolute_percentage_error

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

def analyse_highest_errors(y_test, y_pred, X_test, dp):
    '''
    Function to analyse the data points with the highest errors in the prediction based on the MAPE.

    Parameters:
    y_test: True price values
    y_pred: Predicted price values
    X_test: Test dataset
    dp: DataPipeline object

    Returns:
    df_error: DataFrame with the true and predicted values, error and MAPE
    '''

    pd.set_option('display.float_format', '{:.2f}'.format)
    # Calculate the MAPE
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    # Create a dataframe with the true and predicted values
    df_error = X_test.copy()

    df_error = dp.scaler.inverse_transform(df_error)
    df_error = pd.DataFrame(df_error, columns=X_test.columns)

    df_error['true_price'] = y_test
    df_error['predicted_price'] = y_pred
    df_error['error'] = np.abs(df_error['true_price'] - df_error['predicted_price'])
    df_error['mape'] = df_error['error'] / df_error['true_price'] * 100

    # Sort the dataframe by the highest mape
    df_error = df_error.sort_values(by='mape', ascending=False)

    return df_error