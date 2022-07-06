import glob
from train import load_images_from_csv
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.models import load_model

tf.compat.v1.disable_eager_execution()

img_size = (28, 6)
images = glob.glob('datasets/chickpea/images/*.*')


class Client:

    def __init__(self, path_to_model='models/chickpea_model_1.h5',
                 csv_file='datasets/chickpea/response1.csv'):
        self.model = load_model(path_to_model)
        self.path_to_csv = csv_file
        self.x_train, self.y_train, self.x_test, self.y_test = self.get_xy()

    def get_model(self):
        return self.model

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=32)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

    def get_xy(self):
        data_images, data_labels = load_images_from_csv('datasets/chickpea/images',
                                                        self.path_to_csv
                                                        )
        data_images = data_images / 255.0
        data_labels = data_labels / 365.0

        x_train, x_test, y_train, y_test = train_test_split(
            data_images, data_labels, test_size=0.2, random_state=1234
        )
        return x_train, y_train, x_test, y_test

    def set_parameters(self, mean_weights):
        self.model.set_weights(mean_weights)
