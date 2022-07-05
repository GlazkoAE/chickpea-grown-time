from keras import backend as K
from keras.layers import (BatchNormalization, Conv2D, Dense, Flatten,
                          MaxPooling2D)
from keras.metrics import MeanAbsoluteError as mae
from keras.metrics import MeanAbsolutePercentageError as mape
from keras.metrics import RootMeanSquaredError as rmse
from keras.models import Sequential


def scaled_sigmoid(x, scale_factor=365):
    return scale_factor * K.sigmoid(x)


def build_model(hp):
    regression_model = Sequential()
    regression_model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            padding="same",
            strides=(1, 1),
            input_shape=(6, 28, 3),
            activation=hp.Choice(
                "first_conv2d_activation",
                ["relu", "tanh"],
            ),
        )
    )
    if hp.Boolean("need_batch_norm_after_first_conv2d"):
        regression_model.add(BatchNormalization())
    regression_model.add(MaxPooling2D(pool_size=(2, 2)))

    regression_model.add(
        Conv2D(
            hp.Int(
                "second_conv2d_out_channels",
                min_value=32,
                max_value=64,
                step=32,
            ),
            kernel_size=(3, 3),
            padding="same",
            strides=(1, 1),
            input_shape=(2, 11, 32),
            activation=hp.Choice(
                "second_conv2d_activation",
                ["relu", "tanh"],
            ),
        )
    )
    if hp.Boolean("need_batch_norm_after_second_conv2d"):
        regression_model.add(BatchNormalization())
    regression_model.add(MaxPooling2D(pool_size=(2, 2)))

    regression_model.add(Flatten(name="flatten"))
    regression_model.add(Dense(128, activation="relu"))
    regression_model.add(Dense(11, activation="softmax"))
    regression_model.add(Dense(1, activation=scaled_sigmoid))

    regression_model.compile(
        optimizer="adam", loss="mse", metrics=[rmse(), mape(), mae()]
    )
    regression_model.summary()

    return regression_model
