from argparse import ArgumentParser
from typing import Tuple, List

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
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("MnistSceneEncoder")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--image_size", type=Tuple[int, int, int], default=(1, 128, 128))  # type: ignore
        parser.add_argument("--latent_dim", type=int, default=1024)
        return parent_parser

    def __init__(self, image_size: Tuple[int, int, int] = (1, 128, 128),
                 latent_dim: int = 1024,
                 lr: float = 0.001, **kwargs):
        super().__init__()
        self.step_n = 0
        self.encoder = Encoder(latent_dim=latent_dim, image_size=image_size)
        self.decoder = Decoder(latent_dim=latent_dim, image_size=image_size)
        self.img_dim = image_size
        self.lr = lr
        self.latent_dim = latent_dim
        self.save_hyperparameters()

    def forward(self, x):
        mu, log_var = self.encoder(x)
        return mu

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            # Reconstruction mode
            return mu

    def training_step(self, batch):
        scene, masks, labels = batch

        masks_encoded = []
        mus = []
        log_vars = []

        for i in range(masks.shape[1]):
            mask = masks[:, i]
            mu, log_var = self.encoder(mask)

            z = self.reparameterize(mu, log_var)

            mus.append(mu)
            log_vars.append(log_var)

            masks_encoded.append(z)

        # collect mask vectors to one
        mu: torch.Tensor = torch.stack(mus, dim=1)
        log_var: torch.Tensor = torch.stack(log_vars, dim=1)
        z: torch.Tensor = torch.stack(masks_encoded, dim=1)

        # multiply by zero empty pictures
        zeroing: torch.Tensor = labels.expand(z.size())
        divider = torch.sum(labels, dim=1)
        divider = divider.expand(-1, self.latent_dim)

        # divide by number of actual images
        mu = torch.sum(mu * zeroing, dim=1) / divider
        log_var = torch.sum(log_var * zeroing, dim=1) / divider
        z = torch.sum(z * zeroing, dim=1) / divider

        # reconstruct from sum vector
        reconstruction = self.decoder(z)

        # calculate loss
        loss = self.loss_f(reconstruction, scene, mu, log_var)
        # mu, log_var = self.encoder(scene)
        # z = self.reparameterize(mu, log_var)

        # l = torch.nn.BCELoss(reduction='sum')
        # lo = l(reconstruction, scene)
        # img = transform_to_image(scene[0])
        # plt.imshow(img)
        # plt.show()
        # img = transform_to_image(reconstruction[0])
        # plt.imshow(img)
        # plt.show()
        # if torch.any(scene < 0.) or torch.any(scene > 1.):
        #     lol = scene.detach().cpu().numpy()
        #     print("Done")

        # log training process
        self.log("combined_loss", loss[0], prog_bar=True)
        self.log("Reconstruct", loss[1], prog_bar=True)
        self.log("KLD", loss[2], prog_bar=True)
        self.logger.experiment.add_image('Target', scene[0], dataformats='CHW', global_step=self.step_n)
        self.logger.experiment.add_image('Reconstruction', reconstruction[0], dataformats='CHW',
                                         global_step=self.step_n)
        self.step_n += 1

        return loss[0]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def loss_f(reconstruction, x, mu, log_var):
        mse_loss = torch.nn.BCELoss(reduction='sum')
        mse = mse_loss(reconstruction, x)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return mse + kld, mse, kld
