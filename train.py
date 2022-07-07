import argparse
import logging
import os

import keras.models
import wandb

import cv2 as cv
import keras_tuner as kt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.utils import set_random_seed
from wandb.keras import WandbCallback

from model import build_model


def load_images_from_folder(folder: str):
    """
    Load images and labels from folder with images for vigna data.
    Add zeros as last row for match CNN's input shape (6, 28, 3)
    :param folder: path to labeled images folder
    """

    images = []
    days = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            img = np.asarray(img)
            zeros_shape = (1, img.shape[1], img.shape[2])
            img = np.r_[img, np.zeros(zeros_shape, dtype=np.float32)]
            aio_plant = filename.split("_")
            flowering_time = aio_plant[2].split(".")[0]
            images.append(img.astype(np.float32))
            days.append(int(flowering_time))
    return np.asarray(images), np.asarray(days)


def load_images_from_csv(folder: str, annotations: str):
    """
    Load images from folder and labels from csv file for chickpea data.
    :param annotations: path to annotation csv file
    :param folder: path to unlabeled images folder
    """

    images = []
    df = pd.read_csv(annotations, header=None)
    days = df.iloc[:, 1].values
    for image_name in df.iloc[:, 0].values:
        img = cv.imread(os.path.join(folder, str(image_name)))
        images.append(img)
    return np.asarray(images), days


def train(args):
    np.random.seed(args.rs)
    set_random_seed(args.rs)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    K.set_image_data_format("channels_last")

    if args.annotation_file is None:
        data_images, data_labels = load_images_from_folder(args.images_dir)
    else:
        data_images, data_labels = load_images_from_csv(
            args.images_dir, args.annotation_file
        )
    data_images = data_images / 255.0
    data_labels = data_labels / 365.0

    train_images, test_images, train_predicts, test_labels = train_test_split(
        data_images, data_labels, test_size=0.2, random_state=args.rs
    )

    if args.base_model is None:
        tuner = kt.RandomSearch(
            build_model,
            objective="val_loss",
            max_trials=5,
            seed=args.rs,
            overwrite=True,
        )
        tuner.search(
            train_images,
            train_predicts,
            epochs=args.tuner_epochs,
            batch_size=args.tuner_bs,
            validation_split=0.2,
        )
        best_model = tuner.get_best_models()[0]
    else:
        best_model = load_model(args.base_model)

    batch_size = args.bs
    epochs = args.epochs
    verbosity = 1

    if not os.path.isdir("models"):
        os.mkdir("models")

    for num in range(3):
        np.random.seed(args.rs + num)
        set_random_seed(args.rs + num)

        model = best_model

        model_name = args.model_name + '_model_' + str(num + 1)
        model_path = 'models/' + model_name + '.h5'

        wandb_session = wandb.init(
            project='chickpea_grown_predict',
            name=model_name,
            group=args.model_name,
            reinit=True
        )
        mcp_save = ModelCheckpoint(
            model_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=verbosity,
            mode='min'
        )
        reduce_lr_loss = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=7,
            verbose=verbosity,
            epsilon=1e-4,
            mode='min'
        )

        model.fit(
            data_images,
            data_labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbosity,
            validation_split=0.2,
            callbacks=[
                mcp_save,
                early_stopping,
                reduce_lr_loss,
                WandbCallback()
            ],
        )

        # Generate generalization metrics
        model = keras.models.load_model(model_path)
        scores = model.evaluate(test_images, test_labels, verbose=0)
        print(
            f"{model.metrics_names[0]} of {scores[0]}\n"
            f"{model.metrics_names[1]} of {scores[1]}\n"
            f"{model.metrics_names[2]} of {scores[2]}\n"
            f"{model.metrics_names[3]} of {scores[3]}\n"
        )

        wandb_session.finish()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-images_dir", type=str, default="datasets/vigna", help="Directory with images"
    )
    parser.add_argument(
        "-annotation_file", type=str, default=None, help="Path to annotation csv file"
    )
    parser.add_argument(
        "-base_model", type=str, default=None, help="Path to pretrained .h5 model"
    )
    parser.add_argument(
        "-model_name",
        type=str,
        default="best_model",
        help="Path to save trained model",
    )
    parser.add_argument("-rs", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "-k_fold", type=int, default=10, help="Folder number for k-fold"
    )
    parser.add_argument("-bs", type=int, default=128, help="Batch size for train")
    parser.add_argument(
        "-epochs", type=int, default=200, help="Number of epochs for train"
    )
    parser.add_argument(
        "-tuner_bs", type=int, default=128, help="Batch size for tuner random search"
    )
    parser.add_argument(
        "-tuner_epochs",
        type=int,
        default=100,
        help="Number of epochs for tuner random search",
    )
    parser.add_argument(
        "-tuner_max_trials",
        type=int,
        default=10,
        help="Maximum trials for tuner random search",
    )

    arguments = parser.parse_args()

    train(arguments)
