import random
from argparse import ArgumentParser
from typing import Optional, List, Union, Tuple, Iterator

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.dataset.iterable_dataset import MnistIterableDataset
from src.model.scene_vae import MnistSceneEncoder
import pytorch_lightning as pl

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')
program_parser.add_argument("--mnist_download_dir", type=str,
                            default='/home/yessense/PycharmProjects/mnist_scene/mnist_download')
program_parser.add_argument("--dataset_size", type=int, default=10 ** 6)
program_parser.add_argument("--checkpoint_path", type=str,
                            default='/home/yessense/PycharmProjects/mnist_scene/src/model/lightning_logs/version_2/checkpoints/epoch=30-step=121116.ckpt')

# parse input
args = parser.parse_args()


class Experiment:
    def __init__(self, checkpoint_path: str, cuda=True):
        device = 'cuda'
        self.model = self.load_model_from_checkpoint(checkpoint_path)
        self.scenes_gen = self.get_scenes_generator()
        self.scenes = self.get_two_scenes()
        if cuda:
            self.model.to(device)

    def load_model_from_checkpoint(self, checkpoint_path: str) -> MnistSceneEncoder:
        ckpt = torch.load(checkpoint_path)

        hyperparams = ckpt['hyper_parameters']
        state_dict = ckpt['state_dict']

        model = MnistSceneEncoder(**hyperparams)
        model.load_state_dict(state_dict)
        return model

    def get_scenes_generator(self):
        iterable_dataset = MnistIterableDataset(args.mnist_download_dir, args.dataset_size)
        loader = DataLoader(iterable_dataset, batch_size=2, num_workers=1)

        return iter(loader)

    def get_two_scenes(self):
        self.scenes = next(self.scenes_gen)
        return self.scenes

    def process_scenes(self, choices: Optional[List[List[bool]]] = None) -> torch.Tensor:
        if choices is None:
            choices = [[-1 ** (i + j) for i in range(9)] for j in range(2)]
            choices = torch.from_numpy(np.array(choices)).float()

        return self.model.position_latent_operations(self.get_two_scenes(), choices)

    def process_subtraction_scenes(self, choices: Optional[List[bool]]) -> torch.Tensor:
        return self.model.latent_subtraction(self.get_two_scenes(), choices)

    def plot_scenes(self, scenes, result, pos: Union[int, Tuple[int]], operation: str = 'Обмен'):
        names = ['Исходная сц. 1', 'Декодированная сц.1', operation, 'Декодированная сц. 2', 'Исходная сц. 2']
        original_scene1 = self.scenes[0][0]
        original_scene2 = self.scenes[0][1]
        scene1 = scenes[0]
        scene2 = scenes[1]
        result = result[0]
        fig, ax = plt.subplots(1, 5, figsize=(15, 5))
        for i, img in enumerate([original_scene1, scene1, result, scene2, original_scene2]):
            ax[i].imshow(img.detach().cpu().numpy().transpose(1, 2, 0), cmap='gray')
            ax[i].set_axis_off()
            ax[i].set_title(names[i])
        plt.suptitle(f'{operation} #{pos} элем.')
        plt.show()


def create_valid_choices(pos: Union[int, Tuple[int]]) -> torch.Tensor:
    # create one array
    img_1 = [False] * 9
    if isinstance(pos, int):
        img_1[pos] = True
    elif isinstance(pos, Tuple):
        for i in pos:
            img_1[i] = True

    # create second array
    img_2 = [not b for b in img_1]
    choices = [img_1, img_2]
    choices = torch.from_numpy(np.array(choices)).float()
    return choices


def create_not_valid_choices(pos: Union[int, Tuple[int]]) -> torch.Tensor:
    # create one array
    img_1 = [False] * 9
    if isinstance(pos, int):
        img_1[pos] = True
    elif isinstance(pos, Tuple):
        for i in pos:
            img_1[i] = True

    img_2 = [True] * 9
    choices = [img_1, img_2]
    choices = torch.from_numpy(np.array(choices)).float()
    return choices


def create_subtraction_choices(pos: Union[int, Tuple[int]]):
    # create one array
    img_1 = [False] * 9
    if isinstance(pos, int):
        img_1[pos] = True
    elif isinstance(pos, Tuple):
        for i in pos:
            img_1[i] = True
    choices = torch.from_numpy(np.array(img_1)).float()
    return choices



if __name__ == '__main__':
    experiment = Experiment(args.checkpoint_path)

    # ----------------------------------------
    # Exchange 1 number
    # ----------------------------------------
    # for i in range(9):
    #     choices = create_valid_choices(i)
    #     output = experiment.process_scenes(choices=choices)
    #     experiment.plot_scenes(*output, i)

    # ----------------------------------------
    # Exchange x numbers
    # ----------------------------------------
    # for i in range(9):
    #     samples = tuple(random.choices(range(9), k=random.randint(2, 4)))
    #     choice = create_valid_choices(samples)
    #     output = experiment.process_scenes(choices=choices)
    #     experiment.plot_scenes(*output, samples)

    # ----------------------------------------
    # Add number at x pos
    # ----------------------------------------
    # for i in range(9):
    #     pos = 4
    #     choices = create_not_valid_choices(pos)
    #     output = experiment.process_scenes(choices=choices)
    #     experiment.plot_scenes(*output, pos, operation='Добавление')

    # ----------------------------------------
    # Add x number at y pos
    # ----------------------------------------
    # for i in range(9):
    #     samples = tuple(range(i))
    #     choices = create_not_valid_choices(samples)
    #     output = experiment.process_scenes(choices=choices)
    #     experiment.plot_scenes(*output, samples, operation='Добавление')
    #
    # ----------------------------------------
    # Add scene to scene
    # ----------------------------------------
    # for i in range(9):
    #     samples = tuple(range(9))
    #     choices = create_not_valid_choices(samples)
    #     output = experiment.process_scenes(choices=choices)
    #     experiment.plot_scenes(*output, samples, operation='Добавление')
    # ----------------------------------------
    # Subtract 1 items from scene
    # ----------------------------------------
    # for i in range(9):
    #     pos = 4
    #     choices = create_subtraction_choices(pos)
    #     output = experiment.process_subtraction_scenes(choices=choices)
    #     experiment.plot_scenes(*output, pos, operation='Вычитание')
    # ----------------------------------------
    # Subtract x items from scene
    # ----------------------------------------
    # for i in range(9):
    #     pos = 4
    #     choices = create_subtraction_choices(pos)
    #     output = experiment.process_subtraction_scenes(choices=choices)
    #     experiment.plot_scenes(*output, pos, operation='Вычитание')

    print("done")
