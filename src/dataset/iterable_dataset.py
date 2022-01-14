from itertools import cycle, islice

import numpy as np
import torch
import torchvision
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms import ToTensor


class MnistIterableDataset(IterableDataset):
    def __init__(self, mnist_download_dir: str, size: int = 10 ** 5):
        mnist_data = torchvision.datasets.MNIST(mnist_download_dir,
                                                download=True,
                                                train=True,
                                                transform=ToTensor())
        self.data = mnist_data.train_data
        self.labels = mnist_data.train_labels
        self.size = size
        self.y_starts = [i * 42 for i in range(3)]
        self.x_starts = [i * 42 for i in range(3)]

    def sample_generator(self):
        for i in range(self.size):
            scene = np.zeros((1, 128, 128))
            for y_start in self.y_starts:
                for x_start in self.x_starts:
                    img_num = np.random.randint(len(self.data))
                    img = self.data[img_num].numpy()
                    x_shift = np.random.randint(14)
                    y_shift = np.random.randint(14)
                    x = x_start + x_shift
                    y = y_start + y_shift
                    if np.random.randint(10) <= 6:
                        scene[:, y:y + 28, x:x + 28] = img
            yield torch.from_numpy(scene).float(), 'hello'

    def __iter__(self):
        return self.sample_generator()
