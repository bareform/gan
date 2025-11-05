import numpy as np
import torch
import torch.nn as nn

class Generator(nn.Module):
  def __init__(
      self,
      img_size: tuple[int, int, int],
      in_features: list[int],
      latent_dim: int=100
    ) -> None:
    super().__init__()
    self.img_size = img_size
    self.in_features = in_features
    self.latent_dim = latent_dim
    layers = []
    layers.extend([
      nn.Linear(latent_dim, self.in_features[0]),
      nn.BatchNorm1d(self.in_features[0]),
      nn.LeakyReLU(0.2, inplace=True)
    ])
    for i in range(len(self.in_features) - 1):
      layers.extend([
        nn.Linear(self.in_features[i], self.in_features[i + 1]),
        nn.BatchNorm1d(self.in_features[i + 1]),
        nn.LeakyReLU(0.2, inplace=True)
      ])
    layers.extend([
      nn.Linear(self.in_features[-1], int(np.prod(self.img_size))),
      nn.Tanh()
    ])
    self.generator_layers = nn.Sequential(*layers)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    out = self.generator_layers(input)
    out = out.view(out.size(0), *self.img_size)
    return out
