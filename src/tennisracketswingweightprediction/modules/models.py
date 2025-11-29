import torch
from torch import nn
from typing import Type, Optional, List


def _make_dense_block(
    in_features: int, out_features: int, activation_fn: Optional[Type[nn.Module]] = None
) -> nn.Sequential:
    """Build a dense block with Linear -> LayerNorm -> optional activation."""
    layers: List[nn.Module] = [
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
    ]
    if activation_fn is not None:
        layers.append(activation_fn())

    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """Residual block with bottleneck architecture."""

    def __init__(
        self, input_dim: int, hidden_dim: int, activation_fn: Type[nn.Module] = nn.PReLU
    ):
        super().__init__()
        self.block = nn.Sequential(
            _make_dense_block(input_dim, hidden_dim, activation_fn),
            _make_dense_block(hidden_dim, input_dim, None),
        )
        self.final_activation = activation_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out += residual
        return self.final_activation(out)


class Discriminator(nn.Module):
    """Fully connected discriminator network."""

    def __init__(
        self,
        input_length: int,
        hidden_dim: int = 100,
        num_blocks: int = 3,
        activation_fn: Type[nn.Module] = nn.PReLU,
        use_logits: bool = True,
    ):
        """Initialize discriminator.

        Args:
            input_length: Input vector dimension
            hidden_dim: Hidden layer size
            num_blocks: Number of hidden blocks
            activation_fn: Activation function class
            use_logits: Output logits (True) or probabilities (False)
        """
        super().__init__()

        layers: List[nn.Module] = []

        layers.append(_make_dense_block(input_length, hidden_dim, activation_fn))

        for _ in range(num_blocks):
            layers.append(_make_dense_block(hidden_dim, hidden_dim, activation_fn))

        layers.append(nn.Linear(hidden_dim, 1))

        if not use_logits:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResidualDiscriminator(nn.Module):
    """Discriminator using residual blocks."""

    def __init__(
        self,
        input_length: int,
        hidden_dim: int = 100,
        num_blocks: int = 3,
        activation_fn: Type[nn.Module] = nn.PReLU,
        use_logits: bool = True,
    ):
        """Initialize residual discriminator.

        Args:
            input_length: Input vector dimension
            hidden_dim: Hidden layer size
            num_blocks: Number of residual blocks
            activation_fn: Activation function class
            use_logits: Output logits (True) or probabilities (False)
        """
        super().__init__()

        layers: List[nn.Module] = []

        layers.append(_make_dense_block(input_length, hidden_dim, activation_fn))

        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim, hidden_dim, activation_fn))
            layers.append(nn.LayerNorm(hidden_dim))

        layers.append(nn.Linear(hidden_dim, 1))

        if not use_logits:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Generator(nn.Module):
    """Fully connected generator network."""

    def __init__(
        self,
        input_length: int,
        hidden_dim: int = 100,
        num_blocks: int = 3,
        activation_fn: Type[nn.Module] = nn.PReLU,
    ):
        super().__init__()

        layers: List[nn.Module] = []

        layers.append(_make_dense_block(input_length, hidden_dim, activation_fn))

        for _ in range(num_blocks):
            layers.append(_make_dense_block(hidden_dim, hidden_dim, activation_fn))

        layers.append(nn.Linear(hidden_dim, input_length))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResidualGenerator(nn.Module):
    """Generator using residual blocks."""

    def __init__(
        self,
        input_length: int,
        hidden_dim: int = 100,
        num_blocks: int = 2,
        activation_fn: Type[nn.Module] = nn.PReLU,
    ):
        super().__init__()

        layers: List[nn.Module] = []

        layers.append(_make_dense_block(input_length, hidden_dim, activation_fn))
        layers.append(_make_dense_block(hidden_dim, hidden_dim, activation_fn))

        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim, hidden_dim, activation_fn))
            layers.append(nn.LayerNorm(hidden_dim))

        layers.append(nn.Linear(hidden_dim, input_length))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
