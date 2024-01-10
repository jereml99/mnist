from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader

from mnist.data.dataloader import make_training_dataloader
from mnist.models.model import MyAwesomeModel


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=20, help="number of epochs to train for")
@click.option("--batch_size", default=256, help="batch size to use for training")
def train(lr, epochs, batch_size):
    """
    Train a model on the MNIST dataset.
    Args:
        lr: learning rate to use for training
        epochs: number of epochs to train for
        batch_size: batch size to use for training
    """
    print("Training day and night")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(lr)

    model = MyAwesomeModel()
    train_data_loader = make_training_dataloader(batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch in train_data_loader:
            images, labels = batch
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch} Loss: {loss}")

    model_dir = Path(f"models/{model._get_name()}.pt")
    if not model_dir.parent.exists():
        model_dir.parent.mkdir(parents=True)
    torch.save(model, model_dir)

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

    preds, target = [], []
    for batch in train_data_loader:
        x, y = batch
        probs = model(x)
        preds.append(probs.argmax(dim=-1))
        target.append(y.detach())

    target = torch.cat(target, dim=0)
    preds = torch.cat(preds, dim=0)

    report = classification_report(target, preds)
    with open("classification_report.txt", "w") as outfile:
        outfile.write(report)
    confmat = confusion_matrix(target, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confmat,
    )
    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    train()
