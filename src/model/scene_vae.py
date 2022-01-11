from typing import Tuple

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

from src.model.encoder import Encoder


class MnistScene(pl.LightningModule):
    def __init__(self, image_size: Tuple[int, int, int] = (1, 128, 128),
                 latent_dim: int = 1024):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)

    def forward(self, x):
        embedding = self.encoder(x)

    def training_step(self, batch) -> STEP_OUTPUT:
        x, y = batch
        return x



