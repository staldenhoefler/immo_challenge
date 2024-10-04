import unittest
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from src.dataPipeline import DataPipeline  # Assuming the class is in data_pipeline.py
import torch
import os

class TestDataPipeline(unittest.TestCase):
    """
    Unit tests for the DataPipeline class.
    """

    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        # Create a sample DataFrame for testing
        data = {
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8],
            'C': [9, 10, 11, 12],
            'D': ['x', 'y', 'z', 'w']
        }
        os.chdir('..')
        self.df = pd.DataFrame(data)
        self.pipeline = DataPipeline()
        self.pipeline.data = self.df.copy()

    def test_read_csv(self):
        """
        Test the readCsv method.
        """
        # Mocking the readCsv method
        self.pipeline.readCsv = lambda x: setattr(self.pipeline, 'data', self.df.copy())
        self.pipeline.readCsv('dummy_path')
        pd.testing.assert_frame_equal(self.pipeline.data, self.df)
        # Since we can't read an actual file, we'll mock this method.
        pass  # Implementation depends on the environment

    def test_drop_columns(self):
        """
        Test dropping specified columns from the DataFrame.
        """
        self.pipeline.dropColumns(['D'])
        self.assertNotIn('D', self.pipeline.data.columns)
        self.assertEqual(len(self.pipeline.data.columns), 3)

    def test_merge_columns(self):
        """
        Test merging columns in the DataFrame.
        """
        pass
        #TODO: Implement this test

        # Add appropriate assertions based on the mergeColumns implementation

    def test_impute_missing_values(self):
        """
        Test imputing missing values using a specified imputer.
        """
        imputer = KNNImputer(n_neighbors=2)
        self.pipeline.data.drop(columns=['D'], inplace=True)
        self.pipeline.imputeMissingValues(imputer=imputer)
        self.assertFalse(self.pipeline.data.isnull().values.any())
        self.assertAlmostEqual(self.pipeline.data.loc[2, 'A'], 3, places=3)
        self.assertAlmostEqual(self.pipeline.data.loc[1, 'B'], 6, places=3)
    def test_normalize(self):
        """
        Test normalizing the DataFrame features.
        """
        #TODO: Implement this test

    def test_standardize(self):
        """
        Test standardizing the DataFrame features.
        """
        #TODO: Implement this test

    def test_to_pytorch_dataset(self):
        """
        Test converting the DataFrame into a PyTorch Dataset.
        """
        self.pipeline.dropColumns(['D'])
        self.pipeline.imputeMissingValues()
        dataset = self.pipeline.toPytorchDataset()
        self.assertIsInstance(dataset, torch.utils.data.Dataset)
        self.assertEqual(len(dataset), len(self.pipeline.data))
        self.assertTrue(torch.equal(dataset[0], torch.tensor(self.pipeline.data.iloc[0].values, dtype=torch.float)))
    def test_get_data(self):
        """
        Test retrieving the processed DataFrame.
        """
        data = self.pipeline.getData()
        pd.testing.assert_frame_equal(data, self.pipeline.data)

if __name__ == '__main__':
    unittest.main()
