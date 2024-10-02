import unittest
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from data_pipeline import DataPipeline  # Assuming the class is in data_pipeline.py
import torch

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
        self.df = pd.DataFrame(data)
        self.pipeline = DataPipeline()
        self.pipeline.data = self.df.copy()

    def test_read_csv(self):
        """
        Test the read_csv method.
        """
        # Since we can't read an actual file, we'll mock this method.
        pass  # Implementation depends on the environment

    def test_drop_columns(self):
        """
        Test dropping specified columns from the DataFrame.
        """
        self.pipeline.drop_columns(['D'])
        self.assertNotIn('D', self.pipeline.data.columns)
        self.assertEqual(len(self.pipeline.data.columns), 3)

    def test_merge_columns(self):
        """
        Test merging columns in the DataFrame.
        """
        # As merge_columns is not implemented, we skip this test.
        pass

    def test_impute_missing_values(self):
        """
        Test imputing missing values using a specified imputer.
        """
        imputer = SimpleImputer(strategy='mean')
        self.pipeline.impute_missing_values(imputer=imputer)
        self.assertFalse(self.pipeline.data.isnull().values.any())

    def test_normalize(self):
        """
        Test normalizing the DataFrame features.
        """
        self.pipeline.drop_columns(['D'])
        self.pipeline.impute_missing_values()
        self.pipeline.normalize()
        # Check if data is normalized (norm of each row is 1)
        norms = np.linalg.norm(self.pipeline.data.values, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-7)

    def test_standardize(self):
        """
        Test standardizing the DataFrame features.
        """
        self.pipeline.drop_columns(['D'])
        self.pipeline.impute_missing_values()
        self.pipeline.standardize()
        # Check if data has mean ~0 and std ~1
        means = self.pipeline.data.mean()
        stds = self.pipeline.data.std()
        np.testing.assert_allclose(means, np.zeros(len(means)), atol=1e-7)
        np.testing.assert_allclose(stds, np.ones(len(stds)), atol=1e-7)

    def test_to_pytorch_dataset(self):
        """
        Test converting the DataFrame into a PyTorch Dataset.
        """
        self.pipeline.drop_columns(['D'])
        self.pipeline.impute_missing_values()
        dataset = self.pipeline.to_pytorch_dataset()
        self.assertIsInstance(dataset, torch.utils.data.Dataset)
        self.assertEqual(len(dataset), len(self.pipeline.data))

    def test_get_data(self):
        """
        Test retrieving the processed DataFrame.
        """
        data = self.pipeline.get_data()
        pd.testing.assert_frame_equal(data, self.pipeline.data)

if __name__ == '__main__':
    unittest.main()
