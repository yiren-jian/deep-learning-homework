import torch
from torch.utils.data import TensorDataset


def load_dataset(dataset_path, mean_subtraction=False, normalization=False):
    """
    Reads the train and validation for training

    Arguments
    ---------
    dataset_path (string): the path to the dataset
    mean_subtraction (boolean): specifies whether to do mean centering or not. Default: False
    normalization (boolean): specifies whether to normalizes the data or not. Default: False

    Returns
    -------
    train_ds (TensorDataset): The input examples and corresponding labels bundled together
    """
    # Load the dataset
    dataset = torch.load(dataset_path)
    features = dataset['features']
    labels = dataset['labels']

    # Do mean-subtraction if specified
    if mean_subtraction:
        mean = torch.mean(features, dim=0)
        features -= mean

    # do normalization if specified
    if normalization:
        var = torch.var(features, dim=0)
        features /= var

    # create tensor dataset ds
    train_ds = TensorDataset(features, labels)

    return train_ds
