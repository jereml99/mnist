from pathlib import Path
import torch
import click
from torch.utils.data import DataLoader
from mnist.data.dataloader import make_training_dataloader
from mnist.models.model import MyAwesomeModel





@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=20, help="number of epochs to train for")
@click.option("--batch_size", default=256, help="batch size to use for training")
def train(lr, epochs, batch_size):
    """Train a model on MNIST."""
    print("Training day and night")
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
    torch.save(model, model_dir)

if __name__ == '__main__':
   train()
