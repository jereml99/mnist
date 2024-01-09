from torch import nn, Tensor

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),  # [B, 784]
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        """Forward pass of the model with error handling."""

        # Check if input is a tensor
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a PyTorch Tensor.")

        # Check if input tensor has at least 2 dimensions
        if x.ndim < 2:
            raise ValueError("Input tensor must have at least 2 dimensions.")

        # Flattening the tensor
        x_flattened = x.flatten(1)

        # Check if the flattened tensor has the correct shape
        if x_flattened.shape[1] != 784:
            raise ValueError("The flattened input tensor must have 784 features in the second dimension.")

        # Applying the model to the flattened tensor
        return self.model(x_flattened)
