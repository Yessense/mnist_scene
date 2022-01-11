from typing import Tuple

import pytorch_lightning as pl
import torch.optim
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

from src.model.encoder import Encoder


class MnistSceneEncoder(pl.LightningModule):
    def __init__(self, image_size: Tuple[int, int, int] = (1, 128, 128),
                 latent_dim: int = 1024):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch) -> STEP_OUTPUT:
        x = batch['masks'][:,1]
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



