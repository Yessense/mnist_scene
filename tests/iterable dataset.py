from itertools import cycle, islice

import numpy as np
import torchvision
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms import ToTensor


class MyIterableDataset(IterableDataset):
    def __init__(self, data, labels, n):
        self.data = data
        self.labels = labels
        self.n = n
        self.stop = 0
        self.y_starts = [i * 42 for i in range(3)]
        self.x_starts = [i * 42 for i in range(3)]

    def sample_generator(self):
        for i in range(self.n):
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
            yield ToTensor(scene)

    def __iter__(self):
        return self.sample_generator()



mnist_download_data_dir = '/home/yessense/PycharmProjects/mnist_scene/mnist_download'
mnist_data = torchvision.datasets.MNIST(mnist_download_data_dir,
                                        download=True,
                                        train=True,
                                        transform=ToTensor())

data = mnist_data.train_data
labels = mnist_data.train_labels

iterable_dataset = MyIterableDataset(data, labels, 15)

loader = DataLoader(iterable_dataset, batch_size=4)

for batch in loader:
    print(batch)
