import argparse

from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils

from src.emnist_data_handler import read_emnist
from src.save_model import save


def build_net(training_data, model_name='model', epochs=10):

    # Initialize data
    (x_train, y_train), (x_test, y_test), mapping = training_data

    # flatten 28*28 images to a 784 vector for each image
    num_pixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], num_pixels)
    x_test = x_test.reshape(x_test.shape[0], num_pixels)

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # create model
    model = Sequential()
    model.add(Dense(784, input_dim=784, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=200, verbose=1)

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
