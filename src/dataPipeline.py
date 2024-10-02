import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer
import torch
from torch.utils.data import Dataset


class DataPipeline:
    """
    A class used to create a data pipeline for preprocessing datasets.
    """

    def __init__(self):
        """
        Initialize the DataPipeline class.
        """
        self.data = None

    def read_csv(self, file_path):
        """
        Read a CSV file into a pandas DataFrame.

        Parameters:
        file_path (str): The path to the CSV file.
        """
        self.data = pd.read_csv(file_path)

    def drop_columns(self, columns):
        """
        Drop specified columns from the DataFrame.

        Parameters:
        columns (list): List of column names to drop.
        """
        self.data.drop(columns=columns, inplace=True)

    def merge_columns(self):
        """
        Merge specified columns in the DataFrame.

        Note: Implementation to be added later.
        """
        pass  # To be implemented later

        self.data['Floor'] = self.data[['Floor', 'detail_responsive#floor', 'Floor_merged']].bfill(axis=1)['Floor']
        self.data['Space extracted'] = self.data[['Space extracted', 'detail_responsive#surface_living']].bfill(axis=1)[
            'Space extracted']
        self.data['Plot_area_unified'] = self.data[['Plot_area_unified', 'detail_responsive#surface_property', 'Plot_area_merged']].bfill(axis=1)['Plot_area_unified']


    def impute_missing_values(self, imputer=SimpleImputer()):
        """
        Impute missing values in the DataFrame using the specified imputer.

        Parameters:
        imputer (sklearn.impute._base.SimpleImputer): An instance of an imputer.
        """
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)

    def normalize(self):
        """
        Normalize the DataFrame features.
        """
        normalizer = Normalizer()
        self.data = pd.DataFrame(normalizer.fit_transform(self.data), columns=self.data.columns)

    def standardize(self):
        """
        Standardize the DataFrame features.
        """
        scaler = StandardScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)

    def to_pytorch_dataset(self):
        """
        Convert the DataFrame into a PyTorch Dataset.

        Returns:
        torch.utils.data.Dataset: The PyTorch dataset.
        """

        class PandasDataset(Dataset):
            def __init__(self, data):
                self.data = torch.tensor(data.values, dtype=torch.float)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        return PandasDataset(self.data)

    def get_data(self):
        """
        Get the processed DataFrame.

        Returns:
        pandas.DataFrame: The processed DataFrame.
        """
        return self.data

    def run_pipeline(self, file_path:str = "../data/immo_data_202208_v2.csv", columns_to_drop:list = [], imputer=SimpleImputer(), normalize_and_standardize:bool = True):
        """
        Run the data pipeline.
        """
        self.read_csv(file_path)

        #TODO: Make dropping of Columns with config file
        self.drop_columns(columns_to_drop)


        self.impute_missing_values(imputer)

        if normalize_and_standardize:
            self.normalize()
            self.standardize()
