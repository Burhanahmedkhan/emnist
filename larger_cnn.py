import argparse
import numpy

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from src.emnist_data_handler import read_emnist
from src.save_model import save

K.set_image_dim_ordering('th')


def build_net(training_data, model_name='model', epochs=10):

    # Initialize data
    (x_train, y_train), (x_test, y_test), mapping = training_data

    # reshape to be [samples][pixels][width][height]
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=200, verbose=1)
    print(model.summary())

    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

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
