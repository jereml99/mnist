from pathlib import Path

import torch
from click.testing import CliRunner

from mnist.train_model import train


def test_train():
    lr = "0.001"
    epochs = "1"
    batch_size = "128"
    runner = CliRunner()
    # Pass arguments as strings
    results = runner.invoke(train, ["--lr", lr, "--epochs", epochs, "--batch_size", batch_size])
    assert results.exit_code == 0

    # Corrected path handling
    model_name = "MyAwesomeModel"
    model_dir = Path(f"models/{model_name}.pt")
    assert model_dir.exists()

    # Remove the file after test
    model_dir.unlink()
