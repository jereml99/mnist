import torch

from mnist.models.model import MyAwesomeModel


def test_model():
    batch_size = 100
    random_input = torch.rand(batch_size, 1, 28, 28)
    model = MyAwesomeModel()
    output = model(random_input)
    assert output.shape == (batch_size, 10)