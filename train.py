import logging

from sklearn.model_selection import KFold, train_test_split

import numpy as np
import os
import cv2 as cv
import keras_tuner as kt
import argparse
from keras import backend as K
from keras.models import load_model
from model import build_model
import pandas as pd


def load_images_from_folder(folder):
    images = []
    days = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            aio_plant = filename.split("_")
            flowering_time = aio_plant[2].split(".")[0]
            images.append(np.asarray(img).astype(np.float32))
            days.append(int(flowering_time))
    return np.asarray(images), np.asarray(days)


def load_images_from_csv(folder, annotations):
    images = []
    df = pd.read_csv(annotations, header=None)
    days = df.iloc[:, 0].values
    for image_name in df.iloc[:, 1].values:
        img = cv.imread(os.path.join(folder, image_name))
        images.append(img)
    return np.asarray(images), days


def train(args):
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    K.set_image_data_format('channels_last')

    if args.annotation_file is None:
        data_images, data_labels = load_images_from_folder(
            args.images_dir)
    else:
        data_images, data_labels = load_images_from_csv(
            args.images_dir, args.annotation_file
        )
    data_images /= 255.0

    train_images, test_images, train_predicts, test_labels = \
        train_test_split(data_images, data_labels, test_size=0.2, random_state=args.rs)

    if args.base_model is None:
        tuner = kt.RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=10,
            seed=args.rs,
            overwrite=True
        )
        tuner.search(train_images, train_predicts,
                     epochs=args.tuner_epochs,
                     batch_size=args.tuner_bs,
                     validation_split=0.2)
        tuner.results_summary()
        best_model = tuner.get_best_models()[0]
    else:
        best_model = load_model(args.base_model)

    num_folds = args.k_fold
    batch_size = args.bs
    epochs = args.epochs
    verbosity = 1
    mape_per_fold = []
    rmse_per_fold = []
    mae_per_fold = []
    loss_per_fold = []

    data_images = data_images / 255.0

    # Define the K-fold Cross Validator
    fold_no = 1
    kfold = KFold(n_splits=num_folds, shuffle=True)
    for tr, valid in kfold.split(train_images, train_predicts):
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        best_model.fit(data_images[tr], data_labels[tr],
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbosity,
                       validation_split=0.2)

        # Generate generalization metrics
        scores = best_model.evaluate(data_images[valid], data_labels[valid], verbose=0)
        print(
            f'Score for fold {fold_no}: '
            f'{best_model.metrics_names[0]} of {scores[0]};'
            f'{best_model.metrics_names[1]} of {scores[1]}%'
            f'{best_model.metrics_names[2]} of {scores[2]}%'
            f'{best_model.metrics_names[3]} of {scores[3]}%'
        )

        if not os.path.isdir('models'):
            os.mkdir('models')

        model_path = os.path.join('models', args.save_model_name + f'_fold_{fold_no}_loss_{round(scores[0])}.h5')
        best_model.save(model_path)

        # if len(loss_per_fold) == 0:
        #     best_model.save(args.save_model_path)
        # else:
        #     if scores[0] < min(loss_per_fold):
        #         best_model.save(args.save_model_path)

        loss_per_fold.append(scores[0])
        rmse_per_fold.append(scores[1])
        mape_per_fold.append(scores[2])
        mae_per_fold.append(scores[3])

        fold_no = fold_no + 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(loss_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1}'
              f' - Loss: {loss_per_fold[i]}'
              f' - Root MSE: {rmse_per_fold[i]}%'
              f' - Mean absolute percentage error: {mape_per_fold[i]}%'
              f' - Mean absolute error: {mae_per_fold[i]}%'
              )
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Root MSE: {np.mean(rmse_per_fold)} (+- {np.std(rmse_per_fold)})')
    print(f'> Mean absolute percentage error: {np.mean(mape_per_fold)} (+- {np.std(mape_per_fold)})')
    print(f'> Mean absolute error: {np.mean(mae_per_fold)} (+- {np.std(mae_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-images_dir", type=str, default="datasets/vigna", help="Directory with images"
    )
    parser.add_argument(
        "-annotation_file", type=str, default=None,
        help="Path to annotation csv file"
    )
    parser.add_argument(
        "-base_model", type=str, default=None, help="Path to pretrained .h5 model"
    )
    parser.add_argument(
        "-save_model_name", type=str, default="best_model", help="Path to save trained model"
    )
    parser.add_argument(
        "-rs", type=int, default=1234, help="Random seed"
    )
    parser.add_argument(
        "-k_fold", type=int, default=10, help="Folder number for k-fold"
    )
    parser.add_argument(
        "-bs", type=int, default=128, help="Batch size for train"
    )
    parser.add_argument(
        "-epochs", type=int, default=200, help="Number of epochs for train"
    )
    parser.add_argument(
        "-tuner_bs", type=int, default=128, help="Batch size for tuner random search"
    )
    parser.add_argument(
        "-tuner_epochs", type=int, default=100, help="Number of epochs for tuner random search"
    )
    parser.add_argument(
        "-tuner_max_trials", type=int, default=10, help="Maximum trials for tuner random search"
    )

    arguments = parser.parse_args()

    train(arguments)
