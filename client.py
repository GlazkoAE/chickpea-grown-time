import wandb
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from train import load_images_from_csv

from keras.models import load_model
from wandb.keras import WandbCallback


class Client:

    def __init__(self,
                 path_to_model,
                 csv_file,
                 images_dir,
                 model_save_path='models/client_model.h5',
                 name='client_model',
                 wandb_group='chickpea_clients'):

        self.name = name
        self.wandb_group = wandb_group
        self.images_dir = images_dir
        self.path_to_csv = csv_file
        self.images, self.labels = self.load_data()
        self.init_train_callbacks(model_save_path)
        self.model = load_model(path_to_model)

        self.init_train_callbacks(model_save_path)

    def get_model(self):
        return self.model

    def get_parameters(self):
        return self.model.get_weights()

    def set_parameters(self, weights):
        self.model.set_weights(weights)

    def train(self, epochs=20, batch_size=64):
        wandb_session = self.start_wandb()
        self.model.fit(
            self.images,
            self.labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_split=0.2,
            callbacks=[
                self.mcp_save,
                self.early_stopping,
                self.reduce_lr_loss,
                WandbCallback()
            ],
        )
        wandb_session.finish()

    def start_wandb(self):
        wandb_session = wandb.init(
            project='chickpea_grown_predict',
            name=self.name,
            group=self.wandb_group,
            reinit=True
        )
        return wandb_session

    def load_data(self):
        data_images, data_labels = load_images_from_csv(self.images_dir,
                                                        self.path_to_csv
                                                        )
        data_images = data_images / 255.0
        data_labels = data_labels / 365.0

        return data_images, data_labels

    def init_train_callbacks(self, model_save_path):
        self.mcp_save = ModelCheckpoint(
            model_save_path,
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            mode='min'
        )
        self.early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=0,
            mode='min'
        )
        self.reduce_lr_loss = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=7,
            verbose=0,
            min_delta=1e-4,
            mode='min'
        )
