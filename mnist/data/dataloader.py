import torch
from torch.utils.data import DataLoader


def make_training_dataloader(batch_size):
    train_set = torch.utils.data.TensorDataset(torch.load("data/processed/train_images.pt"), torch.load("data/processed/train_target.pt"))

    return DataLoader(train_set, batch_size=batch_size, shuffle=True)