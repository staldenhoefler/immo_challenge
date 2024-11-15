from torch.utils.data import Dataset
import torch


# Custom Dataset class
class DataFrameDataset(Dataset):
    '''
    Custom dataset class for PyTorch
    '''


    def __init__(self, dataframe, featureColumns, labelColumn, transform=None, yTransform=None):
        self.dataframe = dataframe
        self.features = dataframe[featureColumns].values
        self.labels = dataframe[labelColumn].values
        self.transform = transform
        self.yTransform = yTransform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)

        if self.yTransform:
            y = self.yTransform(y)

        return x, y