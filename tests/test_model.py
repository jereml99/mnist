import torch
import pytest

from mnist.models.model import MyAwesomeModel


def test_model():
    batch_size = 100
    random_input = torch.rand(batch_size, 1, 28, 28)
    model = MyAwesomeModel()
    output = model(random_input)
    assert output.shape == (batch_size, 10), "Output shape is incorrect"  # Check if the output shape is correct


def test_model_with_bad_input_type():
    model = MyAwesomeModel()
    with pytest.raises(TypeError):
        model("bad_input")


def test_model_with_input_less_than_2_dimensions():
    model = MyAwesomeModel()
    bad_input = torch.rand(784)
    with pytest.raises(ValueError):
        model(bad_input)


def test_model_with_flattened_input_incorrect_shape():
    model = MyAwesomeModel()
    bad_input = torch.rand(100, 784, 2, 1)
    with pytest.raises(ValueError):
        model(bad_input)