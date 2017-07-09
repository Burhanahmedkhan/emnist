import argparse
import numpy as np

from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K

from src.emnist_data_handler import read_emnist
from src.save_model import save

K.set_image_dim_ordering('th')


def build_net(training_data, model_name, epochs=10):

    # Initialize data
    (x_train, y_train), (x_test, y_test), mapping = training_data
    input_shape = (1, 28, 28)
    nb_classes = len(mapping)

    # Reshape arrays to (None, 28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

    # Hyperparameters
    batch_size = 256
    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    model = Sequential()
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            padding='valid',
                            input_shape=input_shape,
                            activation='relu'))
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    print(model.summary())

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-score[1]*100))

    save(model, mapping, model_name)
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='Path to .mat file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train on')
    parser.add_argument('-m', '--model', type=str, help='Model name')
    args = parser.parse_args()

    training_data = read_emnist(args.file)
    model = build_net(training_data, args.model, epochs=args.epochs)
