import os

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from mnist.data.dataloader import make_training_dataloader


def get_intermediate_output(model, data):
    intermediate_output = []
    for images, labels in data:
        # Forward pass through the model
        output = model(images)

        # Get the intermediate output
        intermediate_output.append(output.detach().numpy())

    intermediate_output = np.concatenate(intermediate_output, axis=0)
    return intermediate_output


@click.command()
@click.argument("model_checkpoint")
def visualize(model_checkpoint):
    # Load the pre-trained model
    model = torch.load(model_checkpoint)
    model.eval()

    # Load the training data
    data = make_training_dataloader(256)

    # Get the intermediate representation of the data
    intermediate_output = get_intermediate_output(model, data)

    # Use t-SNE to reduce the dimensionality to 2D
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(intermediate_output)

    # Plot the results
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.title("t-SNE visualization of intermediate features")

    # Save the figure
    if not os.path.exists("reports/figures"):
        os.makedirs("reports/figures")
    plt.savefig("reports/figures/tsne_visualization.png")


if __name__ == "__main__":
    visualize()
