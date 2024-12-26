import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans
import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import pgeocode
import os
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm


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
        self.scaler = None
        self.kmeans = None

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
        columns_that_exist = []
        for column in columns:
            if column in self.data.columns:
                columns_that_exist.append(column)
        self.data.drop(columns=columns_that_exist, inplace=True)

    def mergeColumns(self, clusterGroups=50):
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

        def cleanAndFillRooms(data):
            # Entfernt "m²", "rm" und "r" aus der "rooms"-Spalte
            data["rooms"] = data["rooms"].str.strip("m²").str.strip("rm").str.strip("r")
            data["No. of rooms:"] = data["No. of rooms:"].fillna(data["rooms"])
            data["No. of rooms:"] = data["No. of rooms:"].astype("float", errors="ignore")
            data["No. of rooms:"] = data["No. of rooms:"].replace(0, pd.NA)
            data["No. of rooms:"] = data["No. of rooms:"].fillna(
                data["details"].str.extract(r'(\d+)\s*rooms?', expand=False)
            )
            data["No. of rooms:"] = data["No. of rooms:"].astype("float", errors="ignore")
            return data

        self.data = cleanAndFillRooms(self.data)

        def cleanAndFillSpace(data):
            if 'Space extracted' in data.columns:
                self.data['Space extracted'] = self.data['Space extracted'].apply(
                    lambda x: x.split(' ')[0] if isinstance(x, str) else x
                )
                self.data['Space extracted'] = self.data['Space extracted'].replace({'\'': ''})

            def extractSpace(detail):
                if isinstance(detail, str) and 'm²' in detail:
                    try:
                        return float(detail.split('m²')[0].split()[-1].strip().replace(',', ''))
                    except ValueError:
                        return np.nan
                return np.nan

            # Fill missing values in "Space extracted" using details
            if "Space extracted" not in data.columns:
                data["Space extracted"] = np.nan

            data["Space extracted"] = data["Space extracted"].fillna(
                data["details"].apply(extractSpace)
            )
            data["Space extracted"] = data["Space extracted"].astype("float")
            return data

        #self.data = cleanAndFillSpace(self.data)

        self.data["Last refurbishment:"] = self.data["Last refurbishment:"].fillna(
            self.data["Year built:"]
        )

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
            postal_list = missing_postal_codes['plz_parsed'].values.astype("str").tolist()
            location_data = nomi.query_postal_code(postal_list)
            df.loc[mask, 'lat'] = location_data['latitude'].values
            df.loc[mask, 'lon'] = location_data['longitude'].values
            return df

        self.data = imputeLonLat(self.data)
        if clusterGroups > 0:
            if self.kmeans is None:
                self.groupLonLats_fit_pred(clusterGroups)
            else:
                self.groupLonLats_pred()

        return self.data

    def cleanData(self, params):
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
            self.data['Plot_area_unified'] = self.data['Plot_area_unified'].astype(float)

        # Example of how to process the Availability column
        if 'Availability' in self.data.columns:
            self.data['Availability'] = self.data['Availability'].apply(
                lambda x: 'Future' if len(str(x).split('.')) > 1 else x
            )

        if 'features' in self.data.columns:
            feature_dummies =  self.data['features'].str.get_dummies(sep='\t')
            self.data = pd.concat([self.data, feature_dummies], axis=1)
            self.data = self.data.drop(columns=['features'])

        if 'detailed_description' in self.data.columns:
            self.add_BERT_features()
            self.data = self.data.drop(columns=['detailed_description'])

        # Remove rows with nan in 'price_cleaned' column
        if 'price_cleaned' in self.data.columns:
            self.data = self.data.dropna(subset=['price_cleaned'])

        # Change datatype of every column except of some to float
        for column in self.data.columns:
            if column not in ['Availability', 'type', 'provider', 'type_unified', 'detailed_description']:
                try:
                    self.data[column] = self.data[column].astype(float)
                except:
                    print(f'Error in column: {column}')
                    break

        # drop dublicated rows
        if 'price_cleaned' in self.data.columns:
            self.data.drop_duplicates(inplace=True)

        # drop rows with price below threshold
        if 'price_cleaned' in self.data.columns:
            price_threshold = params['price_threshold']
            price_threshold_upper = params['price_threshold_upper']
            self.data = self.data[self.data['price_cleaned'] > price_threshold]
            self.data = self.data[self.data['price_cleaned'] < price_threshold_upper]

    def add_BERT_features(self):
        # Load model and tokenizer
        model_name = "bert-base-german-cased"
        model = BertForSequenceClassification.from_pretrained("./regression_model")
        tokenizer = BertTokenizer.from_pretrained(model_name)

        # Set model to evaluation mode
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Prepare the descriptions
        descriptions = self.data['detailed_description'].fillna("").tolist()

        # Tokenize and predict in batches
        batch_size = 16
        bert_features = []

        # Use tqdm to add a progress bar
        for i in tqdm(range(0, len(descriptions), batch_size), desc="Processing BERT Features"):
            batch_texts = descriptions[i:i + batch_size]
            tokenized = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=64,
                return_tensors="pt"
            )
            tokenized = {key: value.to(device) for key, value in tokenized.items()}

            # Generate predictions
            with torch.no_grad():
                predictions = model(**tokenized).logits
                predictions = predictions.squeeze().cpu().numpy()
                if predictions.ndim == 0:
                    bert_features.append(predictions)
                else:
                    bert_features.extend(predictions)

        self.data['bert_feature'] = bert_features



    def imputeMissingValues(self, imputer=SimpleImputer()):
        """
        Impute missing values in the DataFrame using the specified imputer.

        Parameters:
        imputer (sklearn.impute._base.SimpleImputer): An instance of an imputer.
        """
        columns_to_drop = ['price_cleaned', 'type', 'type_unified', 'detailed_description']
        columns_to_impute = [col for col in self.data.columns if col not in columns_to_drop]
        self.data[columns_to_impute] = imputer.fit_transform(self.data[columns_to_impute])

    def normalize(self):
        """
        Normalize the DataFrame features.
        """
        normalizer = Normalizer()
        self.data = pd.DataFrame(normalizer.fit_transform(self.data), columns=self.data.columns)

    def standardize(self):
        """
        Standardize the DataFrame features. Except for the target column.
        """
        if self.scaler is None:
            self.scaler = StandardScaler()
            columns_to_drop = ['price_cleaned']
            columns = [col for col in self.data.columns if col not in columns_to_drop]
            self.data[columns] = self.scaler.fit_transform(self.data[columns])
        else:
            columns_to_drop = ['price_cleaned']
            columns = [col for col in self.data.columns if col not in columns_to_drop]
            self.data[columns] = self.scaler.transform(self.data[columns])


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
                    normalizeAndStandardize:bool = False,
                    columnsToDrop: list = [],
                    get_dummies: bool = True,
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


        current_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(current_dir, '..', 'src', 'params.yaml')

        with open(params_path, 'r', encoding='utf-8') as file:
            params = yaml.safe_load(file)

        if columnsToDrop == []:
            columnsToDrop = params['columns_to_drop_all']

        self.readCsv(filePath)

        self.mergeColumns(params['clusterGroups'])

        self.dropColumns(columnsToDrop)
        self.cleanData(params)
        if get_dummies:
            self.encodeCategoricalFeatures()

        if imputer:
            self.imputeMissingValues(imputer)
        if normalizeAndStandardize:
            #self.normalize()
            self.standardize()

        return self.data

    def encodeCategoricalFeatures(self):
        """
        Encode categorical features in the DataFrame.
        """
        self.data = pd.get_dummies(self.data, drop_first=True)


    def groupLonLats_fit_pred(self, numGroups):
        """
        Group the longitude and latitude values into clusters.

        Parameters:
        numGroups (int): The number of groups to create.
        """

        kmeans = KMeans(n_clusters=numGroups)
        self.data['region_group'] = kmeans.fit_predict(self.data[['lon', 'lat']])
        self.kmeans = kmeans
        self.data = pd.get_dummies(self.data, columns=['region_group'])
        self.data = self.data.drop(columns=['lon', 'lat'])

    def groupLonLats_pred(self):
        self.data['region_group'] = self.kmeans.predict(self.data[['lon', 'lat']])
        self.data = pd.get_dummies(self.data, columns=['region_group'])
        self.data = self.data.drop(columns=['lon', 'lat'])


    def prepare_kaggle_dataset(self, filePath, normalizeAndStandardize=False, imputer=None, columnsToDrop=[], get_dummies=True):
        """
        Prepare the dataset for Kaggle submission.

        Returns:
        pandas.DataFrame: The DataFrame for Kaggle submission.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(current_dir, '..', 'src', 'params.yaml')

        with open(params_path, 'r', encoding='utf-8') as file:
            params = yaml.safe_load(file)

        self.readCsv(filePath)

        if columnsToDrop == []:
            columnsToDrop = params['columns_to_drop_all']

        self.mergeColumns(params['clusterGroups'])

        self.dropColumns(columnsToDrop)
        self.cleanData(params)
        if get_dummies:
            self.encodeCategoricalFeatures()

        if imputer:
            self.imputeMissingValues(imputer)

        if normalizeAndStandardize:
            #self.normalize()
            self.standardize()

        return self.data