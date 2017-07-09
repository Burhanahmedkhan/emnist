import numpy as np

from src.predictor import Predictor
from src.emnist_data_handler import read_emnist


emnist_file = "../emnist-dataset/"

(x_train, y_train), (x_test, y_test), mapping = read_emnist(emnist_file)


def test_model(model_name):
    model = Predictor(model_name)

    correct = 0
    wrong = 0

    for x in range(len(y_test)):
        a = x_train[x]
        a_label = y_train[x]
        a = a.reshape(1, 1, 28, 28)
        if model.predict(a) == chr(int(model.mapping[str(a_label)])):
            correct += 1
        else:
            wrong += 1

    print(model_name + " - accuracy: %.2lf%%" % (100*correct/(correct+wrong)))


# Evaluate models
print(test_model("simple_cnn"))
print(test_model("larger_cnn"))
print(test_model("mnist_nn"))
