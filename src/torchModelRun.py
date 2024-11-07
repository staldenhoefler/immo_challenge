from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.CustomDataset import DataFrameDataset
import yaml
import torch
import time
import wandb


def getDataLoaders(data, y_column:str, batch_size:int, train_val_test_split:list=[0.7, 0.0, 0.3], shufle:list=[True, False, False]):
    """
    Get the data loaders for the model training and evaluation.
    """
    feature_columns = [col for col in data.columns if col != y_column]
    label_column = y_column

    # Create dataset and dataloader
    dataset = DataFrameDataset(data, feature_columns, label_column)

    # Set your desired split ratios
    train_ratio = train_val_test_split[0]
    val_ratio = train_val_test_split[1]
    test_ratio = train_val_test_split[2]

    # Calculate lengths for each split
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size  # Ensures all samples are used

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shufle[0])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shufle[1])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shufle[2])

    return train_loader, val_loader, test_loader

def mean_absolute_percentage_error(preds, targets):
    epsilon = 1e-10  # Small value to avoid division by zero
    return torch.mean(torch.abs((targets - preds) / (targets + epsilon))) * 100


def mape_loss(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred) / y_true))

def getParams():
    with open('src/params.yaml', 'r', encoding='utf-8') as file:
        params = yaml.safe_load(file)
    return params

def train(model, device, epochs, train_loader, test_loader, criterion, optimizer):
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        start_traintime = time.time()
        model.train()
        for i, (data) in enumerate(train_loader, 0):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs.flatten(), labels.flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_loss = running_loss / (i + 1)
        endtime_train = time.time() - start_traintime
        test_mape, train_mape, train_loss, test_loss = evaluate(model, device, test_loader, train_loader,
                                                                        train_loss, endtime_train, criterion)
        print(
            f"Epoch {epoch + 1}, Loss: {train_loss}, Train MAPE: {train_mape}, Test MAPE: {test_mape}")
    print("Finished Training")
    wandb.finish()
    return model

def evaluate(model, device, test_loader, train_loader, train_loss, endtime_train, criterion):
    # Evaluate the model on test_loader
    model.to(device)
    model.eval()

    batches = 0
    test_loss = 0
    test_mape = 0
    starttime_test = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            test_loss += criterion(outputs, labels).item()
            test_mape += mean_absolute_percentage_error(labels, outputs).item()
            batches += 1

    # Calculate average loss and MAPE over all batches
    avg_test_loss = test_loss / batches
    avg_test_mape = test_mape / batches

    batches = 0
    train_mape = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            train_mape += mean_absolute_percentage_error(labels, outputs).item()
            batches += 1

    train_mape = train_mape / batches

    endtime_test = time.time() - starttime_test
    wandb.log(
        {
            "test_mape": avg_test_mape,
            "train_mape": train_mape,
            "train_loss": train_loss,
            "test_loss": avg_test_loss,
            "time_train": endtime_train,
            "time_test": endtime_test,

        }
    )
    return avg_test_mape, train_mape, train_loss, test_loss


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

def run(Model, data, linear_layers):
    """
    Run the model training and evaluation.
    """

    params = getParams()
    y_column = params['y_column']
    batch_sizes = params['batch_size']
    learning_rates = params['learning_rates']
    train_val_test_split = params['train_val_test_split']
    shufle = params['shufle']
    optimizer = params['optimizer']
    loss_function = params['loss_function']
    epochs = params['epochs']

    models = []
    for batch_size in batch_sizes:
        for lr in learning_rates:

            dict = {
                "dataset": "Immo-Normalized",
                "epochs": epochs,
                "linear_layers": linear_layers,
                "learning_rate": lr,
                "architecture": "MLP",
                "batch_size": batch_size,
                "conv_layers": 0,
            }

            wandb_login(dict, name=f'MLP-bs{batch_size}-lr{lr}')

            model = Model()

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=lr)
            criterion = getattr(nn, loss_function)()
            criterion = mean_absolute_percentage_error
            train_loader, val_loader, test_loader = getDataLoaders(data, y_column, batch_size, train_val_test_split, shufle)
            model = train(model, device, epochs, train_loader, test_loader, criterion, optimizer)
            models.append(model.to('cpu'))

    return models





