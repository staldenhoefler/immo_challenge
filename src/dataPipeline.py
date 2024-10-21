import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer
import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import re
import pgeocode


class DataPipeline:
    """
    A class used to create a data pipeline for preprocessing datasets.

    Example usage:
    from src.dataPipeline import DataPipeline
    from sklearn.impute import KNNImputer

    dp = DataPipeline()
    knnImputer = KNNImputer(n_neighbors=10)
    df_cleaned = dp.run_pipeline(imputer=knnImputer)
    """

    def __init__(self):
        """
        Initialize the DataPipeline class.
        """
        self.data = None

    def readCsv(self, filePath):
        """
        Read a CSV file into a pandas DataFrame.

        Parameters:
        file_path (str): The path to the CSV file.
        """
        self.data = pd.read_csv(filePath)

    def dropColumns(self, columns):
        """
        Drop specified columns from the DataFrame.

        Parameters:
        columns (list): List of column names to drop.
        """
        self.data.drop(columns=columns, inplace=True)

    def mergeColumns(self):
        """
        Merge specified columns in the DataFrame.

        Note: Implementation to be added later.
        """

        self.data['Floor'] = self.data[[
            'Floor', 'detail_responsive#floor', 'Floor_merged'
        ]].bfill(axis=1)['Floor']
        self.data['Space extracted'] = self.data[[
            'Space extracted', 'detail_responsive#surface_living'
        ]].bfill(axis=1)['Space extracted']

        self.data['Plot_area_unified'] = self.data[[
            'Plot_area_unified', 'detail_responsive#surface_property',
            'Plot_area_merged'
        ]].bfill(axis=1)['Plot_area_unified']

        self.data['Availability'] = self.data[[
            'Availability', 'detail_responsive#available_from',
            'Availability_merged', 'Verfügbarkeit', 'Disponibilité',
            'Disponibilità'
        ]].bfill(axis=1)['Availability']

        self.data['No. of rooms:'] = self.data[[
            'No. of rooms:', 'rooms'
        ]].bfill(axis=1)['No. of rooms:']

        def extractPlz(address):
            match = re.search(r"\b\d{4}\b", address)
            if match:
                return int(match.group())
            return np.nan

        def imputePlz(df):
            mask = df['plz_parsed'].isna()
            df.loc[mask, 'plz_parsed'] = df.loc[mask, 'address'].apply(extractPlz)
            df['plz_parsed'] = df['plz_parsed'].astype("Int64")
            return df

        self.data = imputePlz(self.data)

        def imputeLonLat(df):
            nomi = pgeocode.Nominatim('ch')
            mask = df['lat'].isna()  # Check for missing latitude values
            missing_postal_codes = df.loc[mask, 'plz_parsed'].reset_index()
            location_data = nomi.query_postal_code(missing_postal_codes['plz_parsed'].values.astype("str").tolist())
            df.loc[mask, 'lat'] = location_data['latitude'].values
            df.loc[mask, 'lon'] = location_data['longitude'].values
            return df

        self.data = imputeLonLat(self.data)
        self.data

    def cleanData(self):
        """
        Clean the data by removing Units and replacing word with its values.
        """
        # Floor column
        if 'Floor' in self.data.columns:
            self.data['Floor'] = self.data['Floor'].replace({
                '1. Basement': '-1',
                '2. Basement': '-2',
                '3. Basement': '-3',
                '4. Basement': '-4',
                'Basement': '-1',
                'GF': '0',
                'Ground floor': '0'
            })
            self.data['Floor'] = self.data['Floor'].apply(
                lambda x: x.split('.')[0] if isinstance(x, str) else x
            )

        # Nutzfläche column
        if 'detail_responsive#surface_usable' in self.data.columns:
            self.data['detail_responsive#surface_usable'] = self.data[
                'detail_responsive#surface_usable'
            ].apply(
                lambda x: x.split(' ')[0] if isinstance(x, str) else x
            )

        # Stockwerksfläche column
        if 'Floor_space_merged' in self.data.columns:
            self.data['Floor_space_merged'] = self.data['Floor_space_merged'].apply(
                lambda x: x.split(' ')[0] if isinstance(x, str) else x
            )

        # Fläche column
        if 'Space extracted' in self.data.columns:
            self.data['Space extracted'] = self.data['Space extracted'].apply(
                lambda x: x.split(' ')[0] if isinstance(x, str) else x
            )
            self.data['Space extracted'] = self.data['Space extracted'].replace({'\'': ''})

        # Plot_area_unified column
        if 'Plot_area_unified' in self.data.columns:
            self.data['Plot_area_unified'] = self.data['Plot_area_unified'].apply(
                lambda x: x.split(' ')[0] if isinstance(x, str) else x
            )
            self.data['Plot_area_unified'] = self.data[
                'Plot_area_unified'
            ].astype(str).str.replace(',', '')

        # Example of how to process the Availability column
        if 'Availability' in self.data.columns:
            self.data['Availability'] = self.data['Availability'].apply(
                lambda x: 'Future' if len(str(x).split('.')) > 1 else x
            )

        # No. of rooms: column
        if 'No. of rooms:' in self.data.columns:
            self.data['No. of rooms:'] = self.data['No. of rooms:'].replace({'\'':''})

        # Remove rows with nan in 'price_cleaned' column
        self.data = self.data.dropna(subset=['price_cleaned'])

        # Change datatype of every column except of some to float
        for column in self.data.columns:
            if column not in ['Availability', 'type', 'provider']:
                #print(f'{column}: {self.data[column].unique()}')
                self.data[column] = self.data[column].astype(float)

        # Remove rows with nan in 'price_cleaned' column

        # rename column
        #self.data.rename(columns={'Floor': 'Stockwerk'}, inplace=True)

        # drop dublicated rows
        self.data.drop_duplicates(inplace=True)




    def imputeMissingValues(self, imputer=SimpleImputer()):
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

    def toPytorchDataset(self):
        """
        Convert the DataFrame into a PyTorch Dataset.

        Returns:
        torch.utils.data.Dataset: The PyTorch dataset.
        """

        class PandasDataset(Dataset):
            """
            A PyTorch Dataset class for a pandas DataFrame.
            """
            def __init__(self, data):
                self.data = torch.tensor(data.values, dtype=torch.float)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        return PandasDataset(self.data)

    def getData(self):
        """
        Get the processed DataFrame.

        Returns:
        pandas.DataFrame: The processed DataFrame.
        """
        return self.data

    def runPipeline(self,
                    filePath:str = "data/immo_data_202208_v2.csv",
                    imputer=SimpleImputer(),
                    normalizeAndStandardize:bool = False
                    ):
        """
        Run the data pipeline.

        Parameters:
        file_path (str): The path to the CSV file.
        columns_to_drop (list): List of column names to drop. Change in config.ini.
        imputer (sklearn.impute._base.SimpleImputer): An instance of an imputer.
        normalize_and_standardize (bool) False: Whether to normalize and standardize the data.

        Returns:
        pandas.DataFrame: The processed DataFrame.
        """

        self.readCsv(filePath)
        self.mergeColumns()

        # Read configuration file
        with open('src/params.yaml', 'r', encoding='utf-8') as file:
            params = yaml.safe_load(file)
        columns_to_drop = params['columns_to_drop_all']

        self.dropColumns(columns_to_drop)
        self.cleanData()
        self.encodeCategoricalFeatures()
        self.imputeMissingValues(imputer)
        if normalizeAndStandardize:
            self.normalize()
            self.standardize()

        return self.data

    def encodeCategoricalFeatures(self):
        """
        Encode categorical features in the DataFrame.
        """
        self.data = pd.get_dummies(self.data)
