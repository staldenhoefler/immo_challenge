from torch.utils.data import Dataset
import torch


# Custom Dataset class
class DataFrameDataset(Dataset):
    def __init__(self, dataframe, feature_columns, label_column):
        self.dataframe = dataframe
        self.features = dataframe[feature_columns].values
        self.labels = dataframe[label_column].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y