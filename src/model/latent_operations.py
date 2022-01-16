from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from src.dataset.iterable_dataset import MnistIterableDataset
from src.model.scene_vae import MnistSceneEncoder
import pytorch_lightning as pl

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')
program_parser.add_argument("--mnist_download_dir", type=str,
                            default='/home/yessense/PycharmProjects/mnist_scene/mnist_download')
program_parser.add_argument("--dataset_size", type=int, default=10 ** 5)

# parse input
args = parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str) -> pl.LightningModule:
    ckpt = torch.load(checkpoint_path)

    hyperparams = ckpt['hyper_parameters']
    state_dict = ckpt['state_dict']

    model = MnistSceneEncoder(**hyperparams)
    model.load_state_dict(state_dict)
    return model


print("Done")

ckpt_path = '/src/model/lightning_logs/version_17/checkpoints/epoch=2-step=1172.ckpt'
model: pl.LightningModule = load_model_from_checkpoint(checkpoint_path=ckpt_path)


def get_two_scenes() -> torch.Tensor:
    iterable_dataset = MnistIterableDataset(args.mnist_download_dir, args.dataset_size)
    loader = DataLoader(iterable_dataset, batch_size=2, num_workers=1)

    scenes = next(iter(loader))

