import os
import pickle

import click
import numpy as np
import torch
from torch.utils.data import DataLoader


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)


@click.command()
@click.argument("model_checkpoint")
@click.argument("data_path")
def predict_model(model_checkpoint, data_path):
    """Predict using a trained model on the given data."""
    print("Predicting...")

    model = torch.load(model_checkpoint)
    model.eval()

    # if os.path.isdir(data_path):
    #     # Load raw images from a folder
    #     # Your code to load images from the folder goes here
    #     # ...
    #     # # Create a dataloader for the loaded images
    #     # test_set = YourDatasetClass(images, labels)  # Replace YourDatasetClass with the appropriate dataset class
    #     # test_data_loader = DataLoader(test_set, batch_size=256, shuffle=True)
    #     raise NotImplementedError("Loading images from a folder is not implemented yet")
    # else:
    #     # Load images from a numpy or pickle file
    #     # Your code to load images from the file goes here
    #     # ...

    #     # Create a dataloader for the loaded images
    #     test_set = YourDatasetClass(images, labels)  # Replace YourDatasetClass with the appropriate dataset class
    #     test_data_loader = DataLoader(test_set, batch_size=256, shuffle=True)

    # with torch.no_grad():
    #      print(predict(model, test_data_loader))
    raise NotImplementedError("Loading images from a numpy or pickle file is not implemented yet")


if __name__ == "__main__":
    predict_model()
