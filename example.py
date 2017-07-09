from src.predictor import Predictor
from src.emnist_data_handler import read_emnist
import numpy as np

emnist_file = "../emnist-dataset/"

predictor = Predictor("larger_cnn")

(x_train, y_train), (x_test, y_test), mapping = read_emnist(emnist_file)


for x in range(100, 200):
    a = x_train[x]
    a_label = y_train[x]
    a = a.reshape(1,1,28,28)
    if predictor.predict(a) == chr(int(mapping[str(a_label)])):
        print(x, "OK")
    else:
        print(x, "NOT OK")
