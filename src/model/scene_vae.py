from typing import Tuple

import pytorch_lightning as pl
import torch.optim
from matplotlib import pyplot as plt
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import nn

from src.dataset.dataset import transform_to_image

torch.set_printoptions(sci_mode=False)

from src.model.decoder import Decoder
from src.model.encoder import Encoder


class MnistSceneEncoder(pl.LightningModule):
    def __init__(self, image_size: Tuple[int, int, int] = (1, 128, 128),
                 latent_dim: int = 1024):

        super().__init__()
        self.step_n = 0
        self.encoder = Encoder(latent_dim=latent_dim, image_size=image_size)
        self.decoder = Decoder(latent_dim=latent_dim, image_size=image_size)
        self.img_dim = image_size

    def forward(self, x):
        x = self.encoder(x)
        return x

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def training_step(self, batch):
        masks = batch['masks']
        scene = batch['scene']

        masks_encoded = []
        mus = []
        logvars = []
        for i in range(masks.shape[1]):
            mu, log_var = self.encoder(masks[:,i])
            mus.append(mu)
            logvars.append(log_var)
            z = self.reparameterize(mu, log_var)
            masks_encoded.append(z)

        mu = sum(mus)
        log_var = sum(logvars)
        z = sum(masks_encoded)

        reconstruction = self.decoder(z)

        loss = self.loss_f(reconstruction, scene, mu, log_var)
        self.log("combined_loss", loss[0], prog_bar=True)
        self.log("Reconstruct", loss[1], prog_bar=True)
        self.log("KLD", loss[2], prog_bar=True)
        self.logger.experiment.add_image('Target', scene[0], dataformats='CHW', global_step=self.step_n)
        self.logger.experiment.add_image('Reconstruction', reconstruction[0], dataformats='CHW', global_step=self.step_n)
        self.step_n += 1
        return loss[0]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss_f(self, reconstruction, x, mu, logvar):
        mse_loss = torch.nn.BCELoss(reduction='sum')
        mse = mse_loss(reconstruction, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld, mse, kld


