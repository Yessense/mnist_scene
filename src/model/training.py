from torch.utils.data import DataLoader

from src.dataset.dataset import MnistScene
from src.model.scene_vae import MnistSceneEncoder
import pytorch_lightning as pl

if __name__ == '__main__':
    mnist_train_data_dir = '/home/yessense/PycharmProjects/mnist_scene/mnist_train'
    latent_dim = 1024
    image_shape = (1, 128, 128)

    # download_and_create_scenes(path_to_download=mnist_download_data_dir,
    #                            path_to_scenes=mnist_train_data_dir)

    data = MnistScene(mnist_train_data_dir)
    data_loader = DataLoader(data,
                             batch_size=10,
                             shuffle=True)

    autoencoder = MnistSceneEncoder(latent_dim=latent_dim, image_size=image_shape)

    trainer = pl.Trainer(gpus=0)
    trainer.fit(autoencoder, data_loader)