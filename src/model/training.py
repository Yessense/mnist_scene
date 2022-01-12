from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from src.dataset.dataset import MnistScene
from src.model.scene_vae import MnistSceneEncoder
import pytorch_lightning as pl

if __name__ == '__main__':
    # parameters
    mnist_train_data_dir = '/home/yessense/PycharmProjects/mnist_scene/mnist_train'
    latent_dim = 1024
    image_shape = (1, 128, 128)

    # data loader
    data = MnistScene(mnist_train_data_dir)
    data_loader = DataLoader(data,
                             batch_size=128,
                             shuffle=True,
                             num_workers=16)

    # model
    autoencoder = MnistSceneEncoder(latent_dim=latent_dim, image_size=image_shape)

    # callbacks
    monitor = 'combined_loss'

    # early stop
    patience = 5
    early_stop_callback = EarlyStopping(monitor=monitor, patience=patience)

    # checkpoint
    save_top_k = 3
    checkpoint_callback = ModelCheckpoint(monitor=monitor, save_top_k=save_top_k)

    # trainer parameters
    profiler = 'simple'  # 'simple'/'advanced'/None
    max_epochs = 100
    gpus = 1

    # trainer
    trainer = pl.Trainer(gpus=gpus,
                         max_epochs=max_epochs,
                         profiler=profiler,
                         callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(autoencoder, data_loader)
