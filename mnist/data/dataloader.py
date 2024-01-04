import torch
from torch.utils.data import DataLoader


def make_training_dataloader(batch_size):
    """
    Create a dataloader for the MNIST training set.
    Args:
        batch_size: batch size to use for training
    Returns:
        DataLoader for the training set
    """
    train_set = torch.utils.data.TensorDataset(torch.load("data/processed/train_images.pt"), torch.load("data/processed/train_target.pt"))

    return DataLoader(train_set, batch_size=batch_size, shuffle=True)