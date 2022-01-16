import torch

from src.model.scene_vae import MnistSceneEncoder
import pytorch_lightning as pl



def load_model_from_checkpoint(checkpoint_path: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path)

    hyperparams = ckpt['hyper_parameters']
    state_dict = ckpt['state_dict']

    model = MnistSceneEncoder(**hyperparams)
    model.load_state_dict(state_dict)
    return model


print("Done")

ckpt_path = '/src/model/lightning_logs/version_17/checkpoints/epoch=2-step=1172.ckpt'
model: pl.LightningModule = load_model_from_checkpoint()
