import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
  def __init__(
      self,
      img_size: tuple[int, int, int],
      in_features: list[int],
      use_spectral_norm: bool=False,
    ) -> None:
    super().__init__()
    self.img_size = img_size
    self.in_features = in_features
    layers = []
    layers.extend([
      spectral_norm(nn.Linear(int(np.prod(self.img_size)), self.in_features[0]))
        if use_spectral_norm else nn.Linear(int(np.prod(self.img_size)), self.in_features[0]),
      nn.LeakyReLU(0.2, inplace=True)
    ])
    for i in range(len(self.in_features) - 1):
      layers.extend([
        spectral_norm(nn.Linear(self.in_features[i], self.in_features[i + 1]))
           if use_spectral_norm else nn.Linear(self.in_features[i], self.in_features[i + 1]),
        nn.LeakyReLU(0.2, inplace=True)
      ])
    layers.extend([
      spectral_norm(nn.Linear(self.in_features[-1], 1))
        if use_spectral_norm else nn.Linear(self.in_features[-1], 1),
      nn.Sigmoid()
    ])
    self.discriminator_layers = nn.Sequential(*layers)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    out = input.view(input.size(0), -1)
    out = self.discriminator_layers(out)
    return out
