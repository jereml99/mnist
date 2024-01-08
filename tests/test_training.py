from pathlib import Path
import torch
from mnist.train_model import train
from click.testing import CliRunner

def test_train():
    lr = 0.001
    epochs = 1
    batch_size = 128
    runner = CliRunner()
    runner.invoke(train, [lr, epochs, batch_size])

    model_dir = Path(f"models/MyAwesomeModel.pt")
    assert model_dir.exists()

    model_dir.unlink()