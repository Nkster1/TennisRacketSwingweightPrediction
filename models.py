import torch.nn.functional as F
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_length, 100),
            nn.LayerNorm(100),
            nn.PReLU(),
            nn.Linear(100, 100),
            nn.LayerNorm(100),
            nn.PReLU(),
            nn.Linear(100, 100),
            nn.LayerNorm(100),
            nn.PReLU(),
            nn.Linear(100, 100),
            nn.LayerNorm(100),
            nn.PReLU(),
            nn.Linear(100, 50),
            nn.LayerNorm(50),
            nn.PReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.model(x)

class ResidualDiscriminator(nn.Module):
    def __init__(self, input_length: int):
        super(ResidualDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_length, 100),
            nn.LayerNorm(100),
            nn.PReLU(),
            ResidualBlock(100),
            nn.LayerNorm(100),
            ResidualBlock(100),
            nn.LayerNorm(100),
            ResidualBlock(100),
            nn.LayerNorm(100),
            nn.Linear(100, 50),
            nn.LayerNorm(50),
            nn.PReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_dimensions):
        super(ResidualBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dimensions, 2*in_dimensions),
            nn.LayerNorm(2*in_dimensions),
            nn.PReLU(),
            nn.Linear(2*in_dimensions, in_dimensions),
            nn.LayerNorm(in_dimensions),
        )


    def forward(self, x):
        out = self.model(x)
        out += x
        return F.relu(out)

class Generator(nn.Module):

    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        dropout = 0.1
        self.model = nn.Sequential(
            nn.Linear(input_length, 100),
            nn.LayerNorm(100),
            nn.PReLU(),
            nn.Linear(100, 200),
            nn.LayerNorm(200),
            nn.PReLU(),
            nn.Linear(200, 100),
            nn.LayerNorm(100),
            nn.PReLU(),
            nn.Linear(100, 100),
            nn.LayerNorm(100),
            nn.PReLU(),
            nn.Linear(100, 50),
            nn.LayerNorm(50),
            nn.PReLU(),
            nn.Linear(50, input_length),
        )

    def forward(self, x):
        return self.model(x)

class ResidualGenerator(nn.Module):

    def __init__(self, input_length: int):
        super(ResidualGenerator, self).__init__()
        dropout = 0.1
        self.model = nn.Sequential(
            nn.Linear(input_length, 100),
            nn.LayerNorm(100),
            nn.PReLU(),
            nn.Linear(100, 200),
            nn.LayerNorm(200),
            nn.PReLU(),
            nn.Linear(200, 100),
            nn.LayerNorm(100),
            ResidualBlock(100),
            nn.LayerNorm(100),
            ResidualBlock(100),
            nn.LayerNorm(100),
            nn.Linear(100, 50),
            nn.LayerNorm(50),
            nn.PReLU(),
            nn.Linear(50, input_length),
        )

    def forward(self, x):
        return self.model(x)
