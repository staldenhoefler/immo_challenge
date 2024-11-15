from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.customDataset import DataFrameDataset
import yaml
import torch
import time
import wandb
import numpy as np


class StandardizeTransform:
    '''
    Standardize the data
    '''
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


class MinMaxTransform:
    '''
    MinMax-Scale the data
    '''
    def __init__(self, min, max):
        self.min = torch.tensor(min, dtype=torch.float32)
        self.max = torch.tensor(max, dtype=torch.float32)

    def __call__(self, x):
        '''
        Returns MinMax scaled data
        '''
        return (x - self.min) / (self.max - self.min)

    def inverse(self, x):
        '''
        Returns inverse of MinMax scaling
        '''
        return x * (self.max - self.min) + self.min

def getDataLoaders(df, yColumn:str, batchSize:int,
                   trainValTestSplit:list=[0.7, 0.0, 0.3],
                   shufle:list=[True, False, False], transform=None):
    """
    Get the data loaders for the model training and evaluation.
    """
    data = df.copy()
    featureColumns = [col for col in data.columns if col != yColumn]
    labelColumn = yColumn

    data[labelColumn] = np.log(data[labelColumn] + 1)
    yTransform = None
    transform = StandardizeTransform(data[featureColumns].mean().values,
                                     data[featureColumns].std().values)
    #y_transform = MinMaxTransform(data[label_column].min(), data[label_column].max())

    # Create dataset and dataloader
    dataset = DataFrameDataset(data, featureColumns, labelColumn, transform=transform)



    # Set your desired split ratios
    trainRatio = trainValTestSplit[0]
    valRatio = trainValTestSplit[1]

    # Calculate lengths for each split
    totalSize = len(dataset)
    trainSize = int(totalSize * trainRatio)
    valSize = int(totalSize * valRatio)
    testSize = totalSize - trainSize - valSize  # Ensures all samples are used

    trainDataset, valDataset, testDataset = random_split(dataset,
                                                            [trainSize, valSize, testSize])

    trainLoader = DataLoader(trainDataset, batchSize=batchSize, shuffle=shufle[0])
    valLoader = DataLoader(valDataset, batchSize=batchSize, shuffle=shufle[1])
    testLoader = DataLoader(testDataset, batchSize=batchSize, shuffle=shufle[2])

    return trainLoader, valLoader, testLoader, transform,  yTransform

def mean_absolute_percentage_error(preds, targets):
    '''
    Returns the mean absolute percentage error between predictions and targets. Is multiplied by 100 to get percentage.
    '''
    epsilon = 1e-10  # Small value to avoid division by zero
    return torch.mean(torch.abs((targets - preds) / (targets + epsilon))) * 100.0


def mapeLoss(yPred, yTrue):
    '''
    Returns the mean absolute percentage error between predictions and targets. This is a custom loss function.
    '''
    epsilon = 1e-10
    return torch.mean(torch.abs((yTrue - yPred) / (yTrue+epsilon)))

def getParams():
    with open('src/params.yaml', 'r', encoding='utf-8') as file:
        params = yaml.safe_load(file)
    return params

def train(model, device, epochs, trainLoader, testLoader,
          criterion, optimizer, scheduler, yTransformer=None):
    model.to(device)
    alpha = 0.0
    for epoch in range(epochs):
        runningLoss = 0.0
        startTraintime = time.time()
        model.train()
        for i, (data) in enumerate(trainLoader, 0):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            penalty = torch.mean(torch.clamp(outputs - 18.0, min=0.0)) ** 2
            loss += penalty * alpha
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
        train_loss = runningLoss / (i + 1)
        endtimeTrain = time.time() - startTraintime
        (testMape, trainMape,
         trainLoss, testLoss) = evaluate(model,
                                           device, testLoader, trainLoader,
                                           trainLoss, endtimeTrain,
                                           criterion, yTransformer)
        scheduler.step(testLoss)
        print(
            f"Epoch {epoch + 1}, Loss: {train_loss}, "
            f"Train MAPE: {trainMape}, Test MAPE: {testMape}")
    print("Finished Training")
    wandb.finish()
    return model

def evaluate(model, device, testLoader, trainLoader,
             trainLoss, endtimeTrain,
             criterion, yTransformer=None):
    # Evaluate the model on test_loader
    model.to(device)
    model.eval()
    alpha = 0.0

    batches = 0
    testLoss = 0
    testMape = 0
    starttimeTest = time.time()
    with torch.no_grad():
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            testLoss += criterion(outputs, labels).item()
            penalty = torch.mean(torch.clamp(outputs - 18.0, min=0.0)) ** 2
            testLoss += penalty * alpha
            outputs = torch.exp(outputs)
            labels = torch.exp(labels)
            testMape += mean_absolute_percentage_error(labels, outputs).item()
            batches += 1

    # Calculate average loss and MAPE over all batches
    avgTestLoss = testLoss / batches
    avgTestMape = testMape / batches

    batches = 0
    trainMape = 0
    with torch.no_grad():
        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs = torch.exp(outputs)
            labels = torch.exp(labels)
            trainMape += mean_absolute_percentage_error(labels, outputs).item()
            batches += 1

    trainMape = trainMape / batches

    endtimeTest = time.time() - starttimeTest
    wandb.log(
        {
            "test_mape": avgTestMape,
            "train_mape": trainMape,
            "train_loss": trainLoss,
            "test_loss": avgTestLoss,
            "time_train": endtimeTrain,
            "time_test": endtimeTest,

        }
    )
    return avgTestMape, trainMape, trainLoss, testLoss


# implement wandb
def wandb_login(dict, name=None):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Immo-Challenge",
        name=name,
        # track hyperparameters and run metadata
        config=dict
    )

def run(Model, data, linearLayers):
    """
    Run the model training and evaluation.
    """

    params = getParams()
    yColumn = params['y_column']
    batchSizes = params['batch_size']
    learningRates = params['learning_rates']
    trainValTestSplit = params['train_val_test_split']
    shufle = params['shufle']
    optimizer = params['optimizer']
    loss_function = params['loss_function']
    epochs = params['epochs']

    models = []
    for batchSize in batchSizes:
        for lr in learningRates:

            dict = {
                "dataset": "Immo-Normalized",
                "epochs": epochs,
                "linear_layers": linearLayers,
                "learning_rate": lr,
                "architecture": "MLP",
                "batch_size": batchSize,
                "conv_layers": 0,
                "columns": data.columns,
            }

            wandb_login(dict, name=f'MLP-bs{batchSize}-lr{lr}')

            model = Model()

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            optimizer_obj = getattr(torch.optim, optimizer)(model.parameters(), lr=lr)
            scheduler = ReduceLROnPlateau(optimizer_obj, mode='min', factor=0.000001, patience=4)
            criterion = getattr(nn, loss_function)()
            (trainLoader, valLoader, testLoader,
             transform, yTransformer) = getDataLoaders(data, yColumn, batchSize, trainValTestSplit, shufle)
            model = train(model, device, epochs, trainLoader,
                          testLoader, criterion, optimizer_obj, scheduler, yTransformer)
            output = {
                "model": model.to('cpu'),
                "y_transformer": yTransformer,
                "transform": transform,
            }
            models.append(output)

    return models